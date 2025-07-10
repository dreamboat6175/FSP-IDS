%% CyberBattleTCSEnvironment.m - 基于CyberBattleSim思想的TCS环境
% =========================================================================
% 描述: 结合CyberBattleSim概念的列控系统环境，包含误报率分析
% =========================================================================

classdef CyberBattleTCSEnvironment < handle
    
    properties
        % 系统架构
        n_stations              % 主站数量
        n_components            % 各主站组件数数组
        total_components        % 总组件数
        component_importance    % 组件重要性向量
        component_station_map   % 组件到主站的映射
        
        % 网络拓扑（CyberBattleSim风格）
        network_topology        % 网络连接矩阵
        node_vulnerabilities    % 节点漏洞信息
        node_credentials        % 节点凭证信息
        
        % 攻击参数（扩展）
        attack_types           % 攻击类型列表（包含"不攻击"）
        attack_severity        % 攻击严重程度
        attack_detection_difficulty % 检测难度
        n_attack_types         % 攻击类型数量
        attack_kill_chain      % 攻击杀伤链阶段
        
        % 资源参数
        resource_types         % 资源类型列表
        resource_effectiveness % 资源效率
        n_resource_types       % 资源类型数量
        total_resources        % 总资源量
        
        % 状态和动作空间
        state_dim             % 状态空间维度
        action_dim_defender   % 防御者动作空间维度
        action_dim_attacker   % 攻击者动作空间维度（包含"不攻击"）
        
        % 环境状态
        current_state         % 当前状态
        attack_history        % 攻击历史记录
        defense_history       % 防御历史记录
        false_positive_history % 误报历史
        false_negative_history % 漏报历史
        time_step            % 时间步
        
        % 性能指标
        true_positives       % 真阳性
        true_negatives       % 真阴性
        false_positives      % 假阳性（误报）
        false_negatives      % 假阴性（漏报）
        
        % 奖励函数权重
        reward_weights       % 复合奖励函数权重
    end
    
    methods
        function obj = CyberBattleTCSEnvironment(config)
            % 构造函数
            
            % 系统架构初始化
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            % 初始化组件和网络拓扑
            obj.initializeComponents();
            obj.initializeNetworkTopology();
            
            % 攻击参数初始化（包含"不攻击"选项）
            obj.attack_types = ['no_attack', config.attack_types];  % 添加"不攻击"
            obj.attack_severity = [0, config.attack_severity];      % "不攻击"严重度为0
            obj.attack_detection_difficulty = [0, config.attack_detection_difficulty];
            obj.n_attack_types = length(obj.attack_types);
            
            % 初始化攻击杀伤链
            obj.initializeKillChain();
            
            % 资源参数初始化
            obj.resource_types = config.resource_types;
            obj.resource_effectiveness = config.resource_effectiveness;
            obj.n_resource_types = length(obj.resource_types);
            obj.total_resources = config.total_resources;
            
            % 初始化奖励函数权重
            obj.initializeRewardWeights(config);
            
            % 计算状态和动作空间维度
            obj.calculateSpaceDimensions();
            
            % 初始化环境状态
            obj.reset();
        end
        
        function initializeComponents(obj)
            % 初始化组件重要性和站点映射
            obj.component_importance = zeros(1, obj.total_components);
            obj.component_station_map = zeros(1, obj.total_components);
            
            idx = 1;
            for s = 1:obj.n_stations
                for c = 1:obj.n_components(s)
                    % 组件重要性基于类型和位置
                    if c <= 2  % 核心组件（如主控制器）
                        obj.component_importance(idx) = 0.8 + 0.2 * rand();
                    elseif c <= 4  % 重要组件（如通信模块）
                        obj.component_importance(idx) = 0.5 + 0.3 * rand();
                    else  % 一般组件
                        obj.component_importance(idx) = 0.2 + 0.3 * rand();
                    end
                    
                    obj.component_station_map(idx) = s;
                    idx = idx + 1;
                end
            end
            
            % 归一化重要性
            obj.component_importance = obj.component_importance / max(obj.component_importance);
        end
        
        function initializeNetworkTopology(obj)
            % 初始化网络拓扑（CyberBattleSim风格）
            
            % 创建网络连接矩阵
            obj.network_topology = zeros(obj.total_components);
            
            % 站内组件全连接
            idx = 1;
            for s = 1:obj.n_stations
                station_components = idx:(idx + obj.n_components(s) - 1);
                for i = station_components
                    for j = station_components
                        if i ~= j
                            obj.network_topology(i, j) = 1;
                        end
                    end
                end
                idx = idx + obj.n_components(s);
            end
            
            % 站间通过核心组件连接
            for s1 = 1:obj.n_stations
                for s2 = (s1+1):obj.n_stations
                    % 连接各站的第一个组件（假设为核心组件）
                    comp1 = sum(obj.n_components(1:s1-1)) + 1;
                    comp2 = sum(obj.n_components(1:s2-1)) + 1;
                    obj.network_topology(comp1, comp2) = 1;
                    obj.network_topology(comp2, comp1) = 1;
                end
            end
            
            % 初始化节点漏洞
            obj.node_vulnerabilities = rand(obj.total_components, 3) * 0.5;  % 3种漏洞类型
            
            % 初始化节点凭证
            obj.node_credentials = randi([0, 1], obj.total_components, 2);  % 2种凭证类型
        end
        
        function initializeKillChain(obj)
            % 初始化攻击杀伤链阶段
            obj.attack_kill_chain = {
                'reconnaissance',    % 侦察
                'weaponization',     % 武器化
                'delivery',          % 投送
                'exploitation',      % 利用
                'installation',      % 安装
                'command_control',   % 命令与控制
                'actions'            % 行动
            };
        end
        
        function initializeRewardWeights(obj, config)
            % 初始化复合奖励函数权重
            if isfield(config, 'reward_weights')
                obj.reward_weights = config.reward_weights;
            else
                % 默认权重
                obj.reward_weights.w_class = 0.6;    % 分类结果权重
                obj.reward_weights.w_cost = 0.2;     % 行动成本权重
                obj.reward_weights.w_process = 0.2;  % 过程导向权重
                
                % 分类奖励的详细权重
                obj.reward_weights.true_positive = 50;   % 正确检测攻击
                obj.reward_weights.true_negative = 10;   % 正确识别无攻击
                obj.reward_weights.false_positive = -5;  % 误报惩罚
                obj.reward_weights.false_negative = -100; % 漏报严重惩罚
            end
        end
        
        function calculateSpaceDimensions(obj)
            % 计算状态和动作空间维度
            
            % 状态空间：组件状态 + 网络状态 + 历史信息
            obj.state_dim = obj.total_components * 5 + 50;  % 扩展状态表示
            
            % 防御者动作：检测决策 + 资源分配
            obj.action_dim_defender = obj.total_components * 2;  % 每个组件：检测/不检测
            
            % 攻击者动作：目标选择 × 攻击类型（包含"不攻击"）
            obj.action_dim_attacker = obj.total_components * obj.n_attack_types + 1;  % +1 for global "no attack"
        end
        
        function state = reset(obj)
            % 重置环境
            obj.time_step = 0;
            obj.attack_history = [];
            obj.defense_history = [];
            obj.false_positive_history = [];
            obj.false_negative_history = [];
            
            % 重置性能指标
            obj.true_positives = 0;
            obj.true_negatives = 0;
            obj.false_positives = 0;
            obj.false_negatives = 0;
            
            obj.current_state = obj.generateInitialState();
            state = obj.current_state;
        end
        
        function state = generateInitialState(obj)
            % 生成初始状态（向量表示）
            state = zeros(1, obj.state_dim);
            
            % 组件状态
            component_states = rand(obj.total_components, 1);
            state(1:obj.total_components) = component_states;
            
            % 网络连接状态
            network_features = sum(obj.network_topology, 2)' / obj.total_components;
            state(obj.total_components+1:2*obj.total_components) = network_features;
            
            % 其他特征
            state(2*obj.total_components+1:end) = rand(1, length(state) - 2*obj.total_components) * 0.1;
        end
        
        function [next_state, reward, done, info] = step(obj, defender_action, attacker_action)
            % 执行一步环境交互
            
            % 解析动作
            [defense_decision, resource_allocation] = obj.parseDefenderAction(defender_action);
            [attack_target, attack_type] = obj.parseAttackerAction(attacker_action);
            
            % 判断是否有实际攻击
            is_attack = (attack_type > 1);  % attack_type=1 表示"不攻击"
            
            % 防御者检测决策
            detection_result = defense_decision(attack_target);
            
            % 计算检测结果（考虑误报和漏报）
            [detected, is_correct, detection_category] = obj.calculateDetectionResult(...
                is_attack, detection_result, resource_allocation, attack_target, attack_type);
            
            % 更新性能指标
            obj.updatePerformanceMetrics(detection_category);
            
            % 计算复合奖励
            reward = obj.calculateCompositeReward(detection_category, is_attack, ...
                                                attack_target, attack_type, defense_decision);
            
            % 更新状态
            obj.updateState(defense_decision, attack_target, attack_type, detected);
            next_state = obj.current_state;
            
            % 记录信息
            info.is_attack = is_attack;
            info.detected = detected;
            info.detection_category = detection_category;
            info.attack_target = attack_target;
            info.attack_type = attack_type;
            info.false_positive_rate = obj.calculateFPR();
            info.false_negative_rate = obj.calculateFNR();
            info.precision = obj.calculatePrecision();
            info.recall = obj.calculateRecall();
            
            % 检查是否结束
            done = (obj.time_step >= 1000);  % 最大步数
            
            obj.time_step = obj.time_step + 1;
        end
        
        function [defense_decision, resource_allocation] = parseDefenderAction(obj, action)
            % 解析防御者动作
            
            % 简化：前半部分是检测决策，后半部分是资源分配
            decision_part = action(1:obj.total_components);
            defense_decision = decision_part > 0.5;  % 二值化
            
            if length(action) > obj.total_components
                resource_part = action(obj.total_components+1:end);
                resource_allocation = softmax(resource_part);
            else
                resource_allocation = ones(obj.total_components, 1) / obj.total_components;
            end
        end
        
        function [attack_target, attack_type] = parseAttackerAction(obj, action)
            % 解析攻击者动作
            
            if action == 1
                % 全局"不攻击"
                attack_target = 1;  % 默认目标
                attack_type = 1;    % "不攻击"类型
            else
                % 解析具体攻击
                action = action - 1;  % 调整索引
                attack_target = mod(action - 1, obj.total_components) + 1;
                attack_type = floor((action - 1) / obj.total_components) + 1;
                attack_type = min(attack_type, obj.n_attack_types);
            end
        end
        
        function [detected, is_correct, category] = calculateDetectionResult(obj, is_attack, detection_result, resource_allocation, target, attack_type)
            % 计算检测结果，包括误报和漏报
            
            if is_attack
                % 有攻击的情况
                if attack_type > 1
                    % 计算检测概率
                    base_prob = 0.4;  % 基础检测概率
                    resource_bonus = resource_allocation(target) * 0.5;
                    difficulty_penalty = obj.attack_detection_difficulty(attack_type) * 0.3;
                    
                    actual_detection_prob = base_prob + resource_bonus - difficulty_penalty;
                    actual_detection_prob = max(0.1, min(0.95, actual_detection_prob));
                    
                    % 实际检测结果
                    actual_detected = rand() < actual_detection_prob;
                    
                    if detection_result && actual_detected
                        category = 'TP';  % 真阳性
                        detected = true;
                        is_correct = true;
                    elseif ~detection_result && ~actual_detected
                        category = 'TN';  % 真阴性（但实际有攻击，这是错误分类）
                        detected = false;
                        is_correct = false;
                    elseif detection_result && ~actual_detected
                        category = 'FP';  % 假阳性（检测了但实际没检测到）
                        detected = true;
                        is_correct = false;
                    else  % ~detection_result && actual_detected
                        category = 'FN';  % 假阴性（漏报）
                        detected = false;
                        is_correct = false;
                    end
                else
                    % attack_type = 1，即"不攻击"
                    is_attack = false;
                end
            end
            
            if ~is_attack
                % 没有攻击的情况
                if detection_result
                    category = 'FP';  % 假阳性（误报）
                    detected = true;
                    is_correct = false;
                else
                    category = 'TN';  % 真阴性
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
                case 'TN'
                    obj.true_negatives = obj.true_negatives + 1;
                case 'FP'
                    obj.false_positives = obj.false_positives + 1;
                    obj.false_positive_history(end+1) = obj.time_step;
                case 'FN'
                    obj.false_negatives = obj.false_negatives + 1;
                    obj.false_negative_history(end+1) = obj.time_step;
            end
        end
        
        function reward = calculateCompositeReward(obj, category, is_attack, target, attack_type, defense_decision)
            % 计算复合奖励函数
            
            % R_class: 分类结果奖励
            switch category
                case 'TP'
                    r_class = obj.reward_weights.true_positive;
                case 'TN'
                    r_class = obj.reward_weights.true_negative;
                case 'FP'
                    r_class = obj.reward_weights.false_positive;
                case 'FN'
                    r_class = obj.reward_weights.false_negative;
                otherwise
                    r_class = 0;
            end
            
            % 考虑组件重要性
            if is_attack && attack_type > 1
                importance_factor = obj.component_importance(target);
                r_class = r_class * (0.5 + 0.5 * importance_factor);
            end
            
            % R_cost: 行动成本惩罚
            n_active_detections = sum(defense_decision);
            r_cost = -0.1 * n_active_detections;  % 每个主动检测的成本
            
            % R_process: 过程导向奖励
            r_process = 0;
            if is_attack && strcmp(category, 'TP')
                % 早期检测奖励（简化：基于攻击类型）
                if attack_type <= 3  % 假设前3种是早期攻击
                    r_process = 20;
                end
            end
            
            % 总奖励
            reward = obj.reward_weights.w_class * r_class + ...
                    obj.reward_weights.w_cost * r_cost + ...
                    obj.reward_weights.w_process * r_process;
        end
        
        function updateState(obj, defense_decision, attack_target, attack_type, detected)
            % 更新环境状态
            
            % 更新历史
            obj.attack_history(end+1, :) = [obj.time_step, attack_target, attack_type, detected];
            obj.defense_history(end+1, :) = [obj.time_step, sum(defense_decision)];
            
            % 生成新状态
            new_state = zeros(1, obj.state_dim);
            
            % 更新组件状态
            if attack_type > 1 && ~detected
                % 未检测到的攻击会改变组件状态
                component_states = obj.current_state(1:obj.total_components);
                component_states(attack_target) = component_states(attack_target) * 0.8;
                new_state(1:obj.total_components) = component_states;
            else
                new_state(1:obj.total_components) = obj.current_state(1:obj.total_components);
            end
            
            % 更新其他状态特征
            new_state(obj.total_components+1:end) = obj.current_state(obj.total_components+1:end) * 0.95 + ...
                                                   randn(1, length(new_state) - obj.total_components) * 0.05;
            
            obj.current_state = new_state;
        end
        
        % 性能指标计算方法
        function fpr = calculateFPR(obj)
            % 计算误报率
            total_negatives = obj.true_negatives + obj.false_positives;
            if total_negatives > 0
                fpr = obj.false_positives / total_negatives;
            else
                fpr = 0;
            end
        end
        
        function fnr = calculateFNR(obj)
            % 计算漏报率
            total_positives = obj.true_positives + obj.false_negatives;
            if total_positives > 0
                fnr = obj.false_negatives / total_positives;
            else
                fnr = 0;
            end
        end
        
        function precision = calculatePrecision(obj)
            % 计算精确率
            predicted_positives = obj.true_positives + obj.false_positives;
            if predicted_positives > 0
                precision = obj.true_positives / predicted_positives;
            else
                precision = 0;
            end
        end
        
        function recall = calculateRecall(obj)
            % 计算召回率
            actual_positives = obj.true_positives + obj.false_negatives;
            if actual_positives > 0
                recall = obj.true_positives / actual_positives;
            else
                recall = 0;
            end
        end
        
        function f1 = calculateF1Score(obj)
            % 计算F1分数
            precision = obj.calculatePrecision();
            recall = obj.calculateRecall();
            if precision + recall > 0
                f1 = 2 * precision * recall / (precision + recall);
            else
                f1 = 0;
            end
        end
        
        function stats = getStatistics(obj)
            % 获取详细统计信息
            stats.total_steps = obj.time_step;
            stats.true_positives = obj.true_positives;
            stats.true_negatives = obj.true_negatives;
            stats.false_positives = obj.false_positives;
            stats.false_negatives = obj.false_negatives;
            stats.fpr = obj.calculateFPR();
            stats.fnr = obj.calculateFNR();
            stats.precision = obj.calculatePrecision();
            stats.recall = obj.calculateRecall();
            stats.f1_score = obj.calculateF1Score();
            stats.accuracy = (obj.true_positives + obj.true_negatives) / ...
                           max(1, obj.true_positives + obj.true_negatives + ...
                               obj.false_positives + obj.false_negatives);
        end
    end
end

%% 辅助函数
function p = softmax(x)
    % Softmax函数
    ex = exp(x - max(x));
    p = ex / sum(ex);
end