%% CyberBattleTCSEnvironment.m - 基于CyberBattleSim思想的TCS环境
% =========================================================================
% 描述: 结合CyberBattleSim概念的列控系统环境，包含误报率分析
% =========================================================================

classdef CyberBattleTCSEnvironment < handle
    
    properties
        % 系统架构
        n_stations
        n_components
        total_components
        component_importance
        component_station_map
        
        % 网络拓扑
        network_topology
        node_vulnerabilities
        node_credentials
        
        % 攻击参数
        attack_types
        attack_severity
        attack_detection_difficulty
        n_attack_types
        attack_kill_chain
        
        % 资源参数
        resource_types
        resource_effectiveness
        n_resource_types
        total_resources
        
        % 状态和动作空间
        state_dim
        action_dim_defender
        action_dim_attacker
        
        % 环境状态
        current_state
        attack_history
        defense_history
        false_positive_history
        false_negative_history
        time_step
        
        % 性能指标
        true_positives
        true_negatives
        false_positives
        false_negatives
        
        % 奖励函数权重
        reward_weights
    end
    
    methods
        function obj = CyberBattleTCSEnvironment(config)
            % 构造函数
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            obj.initializeComponents();
            obj.initializeNetworkTopology();
            
            obj.attack_types = {'no_attack', config.attack_types{:}};
            obj.attack_severity = [0, config.attack_severity(:)'];
            obj.attack_detection_difficulty = [0, config.attack_detection_difficulty(:)'];
            obj.n_attack_types = length(obj.attack_types);
            
            obj.initializeKillChain();
            
            obj.resource_types = config.resource_types;
            obj.resource_effectiveness = config.resource_effectiveness;
            obj.n_resource_types = length(obj.resource_types);
            obj.total_resources = config.total_resources;
            
            obj.initializeRewardWeights(config);
            obj.calculateSpaceDimensions();
            obj.reset();
        end
        
        function initializeComponents(obj)
            obj.component_importance = zeros(1, obj.total_components);
            obj.component_station_map = zeros(1, obj.total_components);
            idx = 1;
            for s = 1:obj.n_stations
                for c = 1:obj.n_components(s)
                    if c <= 2
                        obj.component_importance(idx) = 0.8 + 0.2 * rand();
                    elseif c <= 4
                        obj.component_importance(idx) = 0.5 + 0.3 * rand();
                    else
                        obj.component_importance(idx) = 0.2 + 0.3 * rand();
                    end
                    obj.component_station_map(idx) = s;
                    idx = idx + 1;
                end
            end
            obj.component_importance = obj.component_importance / max(obj.component_importance);
        end
        
        function initializeNetworkTopology(obj)
            obj.network_topology = zeros(obj.total_components);
            idx = 1;
            for s = 1:obj.n_stations
                station_components = idx:(idx + obj.n_components(s) - 1);
                for i = station_components
                    for j = station_components
                        if i ~= j, obj.network_topology(i, j) = 1; end
                    end
                end
                idx = idx + obj.n_components(s);
            end
            
            for s1 = 1:obj.n_stations
                for s2 = (s1+1):obj.n_stations
                    comp1 = sum(obj.n_components(1:s1-1)) + 1;
                    comp2 = sum(obj.n_components(1:s2-1)) + 1;
                    obj.network_topology(comp1, comp2) = 1;
                    obj.network_topology(comp2, comp1) = 1;
                end
            end
            obj.node_vulnerabilities = rand(obj.total_components, 3) * 0.5;
            obj.node_credentials = randi([0, 1], obj.total_components, 2);
        end
        
        function initializeKillChain(obj)
            obj.attack_kill_chain = {'reconnaissance', 'weaponization', 'delivery', 'exploitation', 'installation', 'command_control', 'actions'};
        end
        
        function initializeRewardWeights(obj, config)
            if isfield(config, 'reward_weights')
                obj.reward_weights = config.reward_weights;
            else
                obj.reward_weights.w_class = 0.6;
                obj.reward_weights.w_cost = 0.2;
                obj.reward_weights.w_process = 0.2;
                obj.reward_weights.true_positive = 50;
                obj.reward_weights.true_negative = 10;
                obj.reward_weights.false_positive = -5;
                obj.reward_weights.false_negative = -100;
            end
        end
        
        function calculateSpaceDimensions(obj)
            obj.state_dim = obj.total_components * 5 + 50;
            obj.action_dim_defender = obj.total_components * 2;
            obj.action_dim_attacker = obj.total_components * obj.n_attack_types + 1;
        end
        
        function state = reset(obj)
            obj.time_step = 0;
            obj.attack_history = [];
            obj.defense_history = [];
            obj.false_positive_history = [];
            obj.false_negative_history = [];
            obj.true_positives = 0;
            obj.true_negatives = 0;
            obj.false_positives = 0;
            obj.false_negatives = 0;
            obj.current_state = obj.generateInitialState();
            state = obj.current_state;
        end
        
        function state = generateInitialState(obj)
            state = zeros(1, obj.state_dim);
            state(1:obj.total_components) = rand(1, obj.total_components);
            state(obj.total_components+1:2*obj.total_components) = sum(obj.network_topology, 2)' / obj.total_components;
            state(2*obj.total_components+1:end) = rand(1, length(state) - 2*obj.total_components) * 0.1;
        end
        
        function [next_state, reward, done, info] = step(obj, defender_action, attacker_action)
            [defense_decision, resource_allocation] = obj.parseDefenderAction(defender_action);
            [attack_target, attack_type] = obj.parseAttackerAction(attacker_action);
            
            is_attack = (attack_type > 1);
            detection_result = defense_decision(attack_target);
            
            [detected, ~, detection_category] = obj.calculateDetectionResult(is_attack, detection_result, resource_allocation, attack_target, attack_type);
            
            obj.updatePerformanceMetrics(detection_category);
            reward = obj.calculateCompositeReward(detection_category, is_attack, attack_target, attack_type, defense_decision);
            obj.updateState(defense_decision, attack_target, attack_type, detected);
            next_state = obj.current_state;
            
            info.is_attack = is_attack;
            info.detected = detected;
            info.detection_category = detection_category;
            info.attack_target = attack_target;
            info.attack_type = attack_type;
            
            done = (obj.time_step >= 1000);
            obj.time_step = obj.time_step + 1;
        end
        
        function [defense_decision, resource_allocation] = parseDefenderAction(obj, action)
            defense_decision = zeros(1, obj.total_components);
            if action <= obj.total_components
                defense_decision(action) = 1;
            end
            
            resource_allocation = zeros(1, obj.total_components);
            if any(defense_decision)
                resource_allocation(defense_decision == 1) = 1;
                resource_allocation = resource_allocation / sum(resource_allocation);
            else
                resource_allocation(:) = 1 / obj.total_components;
            end
        end
        
        function [attack_target, attack_type] = parseAttackerAction(obj, action)
            if action == 1
                attack_target = 1;
                attack_type = 1;
            else
                action_idx = action - 1;
                attack_target = mod(action_idx - 1, obj.total_components) + 1;
                attack_type = floor((action_idx - 1) / obj.total_components) + 1;
                attack_type = min(attack_type, obj.n_attack_types);
            end
        end

        function [detected, is_correct, category] = calculateDetectionResult(obj, is_attack, detection_result, resource_allocation, target, attack_type)
            if is_attack
                base_prob = 0.4;
                resource_bonus = resource_allocation(target) * 0.5;
                difficulty_penalty = obj.attack_detection_difficulty(attack_type) * 0.3;
                actual_detection_prob = max(0.1, min(0.95, base_prob + resource_bonus - difficulty_penalty));
                actual_detected = rand() < actual_detection_prob;
                
                if detection_result && actual_detected
                    category = 'TP'; detected = true; is_correct = true;
                else
                    category = 'FN'; detected = false; is_correct = false;
                end
            else % No attack
                if detection_result
                    category = 'FP'; detected = true; is_correct = false;
                else
                    category = 'TN'; detected = false; is_correct = true;
                end
            end
        end
        
        function updatePerformanceMetrics(obj, category)
            switch category
                case 'TP', obj.true_positives = obj.true_positives + 1;
                case 'TN', obj.true_negatives = obj.true_negatives + 1;
                case 'FP', obj.false_positives = obj.false_positives + 1;
                case 'FN', obj.false_negatives = obj.false_negatives + 1;
            end
        end
        
        function reward = calculateCompositeReward(obj, category, is_attack, target, attack_type, defense_decision)
            switch category
                case 'TP', r_class = obj.reward_weights.true_positive;
                case 'TN', r_class = obj.reward_weights.true_negative;
                case 'FP', r_class = obj.reward_weights.false_positive;
                case 'FN', r_class = obj.reward_weights.false_negative;
                otherwise, r_class = 0;
            end
            
            if is_attack && attack_type > 1
                r_class = r_class * (0.5 + 0.5 * obj.component_importance(target));
            end
            
            r_cost = -0.1 * sum(defense_decision);
            r_process = 0;
            if is_attack && strcmp(category, 'TP') && attack_type <= 3, r_process = 20; end
            
            reward = obj.reward_weights.w_class * r_class + obj.reward_weights.w_cost * r_cost + obj.reward_weights.w_process * r_process;
        end
        
        function updateState(obj, defense_decision, attack_target, attack_type, detected)
            obj.attack_history(end+1, :) = [obj.time_step, attack_target, attack_type, detected];
            obj.defense_history(end+1, :) = [obj.time_step, sum(defense_decision)];
            
            new_state = obj.current_state;
            if attack_type > 1 && ~detected
                new_state(attack_target) = new_state(attack_target) * 0.8;
            end
            new_state(obj.total_components+1:end) = obj.current_state(obj.total_components+1:end) * 0.95 + randn(1, length(new_state) - (obj.total_components)) * 0.05;
            obj.current_state = new_state;
        end
    end
end