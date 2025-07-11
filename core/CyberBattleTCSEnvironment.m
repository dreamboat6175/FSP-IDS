%% CyberBattleTCSEnvironment.m - 基于CyberBattleSim思想的TCS环境（修复版）
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
        time_step
        
        % 性能指标（修复：确保正确累积）
        true_positives
        true_negatives
        false_positives
        false_negatives
        
        % 奖励函数权重
        reward_weights
        
        % 连续正确检测计数
        consecutive_correct
        
        % 添加episode内的统计
        episode_stats
    end
    
    methods
        function obj = CyberBattleTCSEnvironment(config)
            % 构造函数
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            obj.initializeComponents();
            obj.initializeNetworkTopology();
            
            % 修复：正确构建attack_types数组（确保类型一致）
            % 确保attack_types是cell数组格式
            if iscell(config.attack_types)
                obj.attack_types = [{'no_attack'}, config.attack_types(:)'];
            elseif ischar(config.attack_types)
                obj.attack_types = {'no_attack', config.attack_types};
            elseif isstring(config.attack_types)
                obj.attack_types = [{'no_attack'}, cellstr(config.attack_types)];
            else
                % 默认情况：转换为cell数组
                obj.attack_types = [{'no_attack'}, cellstr(config.attack_types)];
            end
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
            obj.consecutive_correct = 0;
            
            % 初始化episode统计
            obj.episode_stats = struct();
            obj.episode_stats.tp = 0;
            obj.episode_stats.tn = 0;
            obj.episode_stats.fp = 0;
            obj.episode_stats.fn = 0;
            
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
            % 简化动作空间，避免过大
            obj.state_dim = min(1000, obj.total_components * 5 + 50);
            obj.action_dim_defender = obj.total_components;  % 简化：每个组件一个动作
            obj.action_dim_attacker = obj.total_components * min(obj.n_attack_types, 5) + 1;  % 限制攻击类型
        end
        
        function state = reset(obj)
            obj.time_step = 0;
            obj.attack_history = [];
            obj.defense_history = [];
            
            % 重置性能指标
            obj.true_positives = 0;
            obj.true_negatives = 0;
            obj.false_positives = 0;
            obj.false_negatives = 0;
            obj.consecutive_correct = 0;
            
            % 重置episode统计
            obj.episode_stats.tp = 0;
            obj.episode_stats.tn = 0;
            obj.episode_stats.fp = 0;
            obj.episode_stats.fn = 0;
            
            obj.current_state = obj.generateInitialState();
            state = obj.current_state;
        end
        
        function state = generateInitialState(obj)
            state = zeros(1, obj.state_dim);
            state(1:obj.total_components) = rand(1, obj.total_components);
            state(obj.total_components+1:2*obj.total_components) = sum(obj.network_topology, 2)' / obj.total_components;
            state(2*obj.total_components+1:end) = rand(1, length(state) - 2*obj.total_components) * 0.1;
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_action, attacker_action)
            % 修复：确保动作有效
            defender_action = max(1, min(defender_action, obj.action_dim_defender));
            attacker_action = max(1, min(attacker_action, obj.action_dim_attacker));
            
            % 解析动作
            [defense_decision, resource_allocation] = obj.parseDefenderAction(defender_action);
            [attack_target, attack_type] = obj.parseAttackerAction(attacker_action);
            
            % 确保索引有效
            attack_target = max(1, min(attack_target, obj.total_components));
            attack_type = max(1, min(attack_type, obj.n_attack_types));
            
            % 判断是否有攻击
            is_attack = (attack_type > 1);
            
            % 获取防御决策
            detection_result = defense_decision(attack_target);
            
            % 计算检测结果
            [detected, is_correct, detection_category] = obj.calculateDetectionResult(...
                is_attack, detection_result, resource_allocation, attack_target, attack_type);
            
            % 更新性能指标
            obj.updatePerformanceMetrics(detection_category);
            
            % 计算奖励
            reward_def = obj.calculateImprovedReward(detection_category, is_attack, ...
                attack_target, attack_type, defense_decision, obj.current_state);
            reward_att = -reward_def * 0.8;  % 攻击者奖励
            
            % 更新状态
            obj.updateState(defense_decision, attack_target, attack_type, detected);
            next_state = obj.current_state;
            
            % 准备信息
            info.is_attack = is_attack;
            info.detected = detected;
            info.detection_category = detection_category;
            info.attack_target = attack_target;
            info.attack_type = attack_type;
            info.defense_decision = defense_decision;
            
            obj.time_step = obj.time_step + 1;
        end
        
        function [defense_decision, resource_allocation] = parseDefenderAction(obj, action)
            % 修复：确保返回正确大小的向量
            defense_decision = zeros(1, obj.total_components);
            resource_allocation = zeros(1, obj.total_components);
            
            % 简单策略：防御者选择一个组件进行防御
            if action >= 1 && action <= obj.total_components
                defense_decision(action) = 1;
                resource_allocation(action) = 1;
            else
                % 默认防御第一个组件
                defense_decision(1) = 1;
                resource_allocation(1) = 1;
            end
        end
        
        function [attack_target, attack_type] = parseAttackerAction(obj, action)
            % 修复：确保解析正确
            if action == 1
                % 无攻击
                attack_target = 1;
                attack_type = 1;  % 对应 'no_attack'
            else
                % 有攻击
                action_idx = action - 1;
                % 计算目标组件和攻击类型
                n_attack_types_excluding_no_attack = obj.n_attack_types - 1;
                attack_target = mod(action_idx - 1, obj.total_components) + 1;
                attack_type_idx = floor((action_idx - 1) / obj.total_components);
                attack_type = min(attack_type_idx + 2, obj.n_attack_types);  % +2 因为索引1是no_attack
                
                % 确保在有效范围内
                attack_target = max(1, min(attack_target, obj.total_components));
                attack_type = max(1, min(attack_type, obj.n_attack_types));
            end
        end
        
        function [detected, is_correct, category] = calculateDetectionResult(obj, is_attack, detection_result, resource_allocation, target, attack_type)
            % 修复：改进检测逻辑，使其更合理
            
            if is_attack
                % 有攻击的情况
                % 基础检测概率
                base_prob = 0.5;  % 提高基础检测率
                
                % 资源加成
                resource_bonus = resource_allocation(target) * 0.4;
                
                % 攻击难度影响
                difficulty_penalty = obj.attack_detection_difficulty(attack_type) * 0.2;
                
                % 最终检测概率
                actual_detection_prob = max(0.2, min(0.95, base_prob + resource_bonus - difficulty_penalty));
                
                % 实际是否检测到
                actual_detected = rand() < actual_detection_prob;
                
                if detection_result && actual_detected
                    % 正确检测到攻击
                    category = 'TP';
                    detected = true;
                    is_correct = true;
                elseif detection_result && ~actual_detected
                    % 报告了攻击但实际没检测到（仍算作误报）
                    category = 'FP';
                    detected = true;
                    is_correct = false;
                elseif ~detection_result && actual_detected
                    % 实际检测到但没报告（算作漏报）
                    category = 'FN';
                    detected = false;
                    is_correct = false;
                else
                    % 没有报告也没有检测到（算作漏报）
                    category = 'FN';
                    detected = false;
                    is_correct = false;
                end
            else
                % 无攻击的情况
                if detection_result
                    % 误报
                    category = 'FP';
                    detected = true;
                    is_correct = false;
                else
                    % 正确识别无攻击
                    category = 'TN';
                    detected = false;
                    is_correct = true;
                end
            end
        end
        
        function updatePerformanceMetrics(obj, category)
            % 更新性能指标
            switch category
                case 'TP'
                    obj.true_positives = obj.true_positives + 1;
                    obj.episode_stats.tp = obj.episode_stats.tp + 1;
                case 'TN'
                    obj.true_negatives = obj.true_negatives + 1;
                    obj.episode_stats.tn = obj.episode_stats.tn + 1;
                case 'FP'
                    obj.false_positives = obj.false_positives + 1;
                    obj.episode_stats.fp = obj.episode_stats.fp + 1;
                case 'FN'
                    obj.false_negatives = obj.false_negatives + 1;
                    obj.episode_stats.fn = obj.episode_stats.fn + 1;
            end
        end
        
        function reward = calculateImprovedReward(obj, category, is_attack, target, attack_type, defense_decision, state)
            % 改进的多层次奖励函数
            
            % 基础分类奖励
            switch category
                case 'TP'
                    r_class = 50;  % 正确检测攻击
                case 'TN'
                    r_class = 10;  % 正确识别正常
                case 'FP'
                    r_class = -5;  % 误报（降低惩罚）
                case 'FN'
                    r_class = -100; % 漏报（严重惩罚）
                otherwise
                    r_class = 0;
            end
            
            % 重要性加权
            if is_attack && attack_type > 1
                importance_weight = 0.5 + 0.5 * obj.component_importance(target);
                r_class = r_class * importance_weight;
            end
            
            % 成本奖励
            r_cost = -0.05 * sum(defense_decision); % 降低成本惩罚
            
            % 过程奖励
            r_process = 0;
            
            % 1. 接近正确防御位置的奖励
            if is_attack && attack_type > 1
                defended_positions = find(defense_decision);
                if ~isempty(defended_positions)
                    distance_to_target = abs(defended_positions - target);
                    min_distance = min(distance_to_target);
                    if min_distance == 0
                        r_process = r_process + 20; % 精确防御
                    elseif min_distance <= 2
                        r_process = r_process + 10; % 接近目标
                    elseif min_distance <= 5
                        r_process = r_process + 5;  % 大致接近
                    end
                end
            end
            
            % 2. 防御覆盖率奖励
            coverage = sum(defense_decision) / obj.total_components;
            if coverage > 0.1 && coverage < 0.3
                r_process = r_process + 5; % 合理的覆盖率
            end
            
            % 3. 连续正确检测奖励
            if strcmp(category, 'TP') || strcmp(category, 'TN')
                obj.consecutive_correct = obj.consecutive_correct + 1;
                if obj.consecutive_correct > 5
                    r_process = r_process + obj.consecutive_correct * 2;
                end
            else
                obj.consecutive_correct = 0;
            end
            
            % 4. 基于状态的奖励（检测高风险状态）
            state_risk = obj.evaluateStateRisk(state);
            if state_risk > 0.7 && any(defense_decision)
                r_process = r_process + 15; % 在高风险时防御
            end
            
            % 组合奖励
            reward = obj.reward_weights.w_class * r_class + ...
                     obj.reward_weights.w_cost * r_cost + ...
                     obj.reward_weights.w_process * r_process;
            
            % 奖励归一化
            reward = obj.normalizeReward(reward);
        end
        
        function risk_level = evaluateStateRisk(obj, state)
            % 评估当前状态的风险级别
            
            % 提取状态特征
            component_states = state(1:obj.total_components);
            network_features = state(obj.total_components+1:2*obj.total_components);
            
            % 计算风险指标
            risk_indicators = 0;
            
            % 1. 组件状态异常
            abnormal_components = sum(component_states < 0.3);
            if abnormal_components > obj.total_components * 0.2
                risk_indicators = risk_indicators + 0.3;
            end
            
            % 2. 网络连接异常
            high_connections = sum(network_features > 0.8);
            if high_connections > obj.total_components * 0.3
                risk_indicators = risk_indicators + 0.3;
            end
            
            % 3. 状态熵（混乱程度）
            state_entropy = -sum(component_states .* log(component_states + 1e-6));
            normalized_entropy = state_entropy / (obj.total_components * log(2));
            if normalized_entropy > 0.7
                risk_indicators = risk_indicators + 0.4;
            end
            
            risk_level = min(1, risk_indicators);
        end
        
        function normalized_reward = normalizeReward(obj, reward)
            % 动态奖励归一化
            persistent reward_history;
            persistent history_size;
            
            if isempty(reward_history)
                reward_history = [];
                history_size = 1000;
            end
            
            reward_history(end+1) = reward;
            if length(reward_history) > history_size
                reward_history(1) = [];
            end
            
            if length(reward_history) > 10
                mean_reward = mean(reward_history);
                std_reward = std(reward_history);
                normalized_reward = (reward - mean_reward) / (std_reward + 1e-6);
                % 限制范围
                normalized_reward = max(-3, min(3, normalized_reward));
            else
                normalized_reward = reward / 100; % 简单归一化
            end
        end
        
        function updateState(obj, defense_decision, attack_target, attack_type, detected)
            obj.attack_history(end+1, :) = [obj.time_step, attack_target, attack_type, detected];
            obj.defense_history(end+1, :) = [obj.time_step, sum(defense_decision)];
            
            new_state = obj.current_state;
            if attack_type > 1 && ~detected
                new_state(attack_target) = new_state(attack_target) * 0.8;
            end
            new_state(obj.total_components+1:end) = obj.current_state(obj.total_components+1:end) * 0.95 + ...
                                                    randn(1, length(new_state) - obj.total_components) * 0.05;
            obj.current_state = new_state;
        end
        
        function metrics = getPerformanceMetrics(obj)
            % 返回分类性能指标
            metrics.true_positives = obj.true_positives;
            metrics.true_negatives = obj.true_negatives;
            metrics.false_positives = obj.false_positives;
            metrics.false_negatives = obj.false_negatives;
            
            % 添加episode统计
            metrics.episode_tp = obj.episode_stats.tp;
            metrics.episode_tn = obj.episode_stats.tn;
            metrics.episode_fp = obj.episode_stats.fp;
            metrics.episode_fn = obj.episode_stats.fn;
            
            % 计算率
            total_positives = obj.true_positives + obj.false_negatives;
            total_negatives = obj.true_negatives + obj.false_positives;
            
            if total_positives > 0
                metrics.tpr = obj.true_positives / total_positives;  % 检测率
            else
                metrics.tpr = 0;
            end
            
            if total_negatives > 0
                metrics.fpr = obj.false_positives / total_negatives;  % 误报率
            else
                metrics.fpr = 0;
            end
            
            % 精确率
            if (obj.true_positives + obj.false_positives) > 0
                metrics.precision = obj.true_positives / (obj.true_positives + obj.false_positives);
            else
                metrics.precision = 0;
            end
            
            % F1分数
            if (metrics.precision + metrics.tpr) > 0
                metrics.f1_score = 2 * metrics.precision * metrics.tpr / (metrics.precision + metrics.tpr);
            else
                metrics.f1_score = 0;
            end
        end
    end
end