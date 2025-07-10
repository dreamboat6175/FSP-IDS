%% ImprovedTCSEnvironment.m - 改进的列控系统环境类
% =========================================================================
% 描述: 优化的列控系统环境，提高学习效率和检测率
% =========================================================================

classdef ImprovedTCSEnvironment < handle
    
    properties
        % 系统架构
        n_stations              % 主站数量
        n_components            % 各主站组件数数组
        total_components        % 总组件数
        component_importance    % 组件重要性向量
        component_station_map   % 组件到主站的映射
        
        % 攻击参数
        attack_types           % 攻击类型列表
        attack_severity        % 攻击严重程度
        attack_detection_difficulty % 检测难度
        n_attack_types         % 攻击类型数量
        
        % 资源参数
        resource_types         % 资源类型列表
        resource_effectiveness % 资源效率
        n_resource_types       % 资源类型数量
        total_resources        % 总资源量
        
        % 状态和动作空间（简化版）
        state_dim             % 状态空间维度（减小到1000）
        action_dim_defender   % 防御者动作空间维度（简化）
        action_dim_attacker   % 攻击者动作空间维度
        
        % 环境状态
        current_state         % 当前状态
        attack_history        % 攻击历史记录
        defense_history       % 防御历史记录
        time_step            % 时间步
        
        % 新增：简化的资源分配策略
        simplified_actions    % 预定义的资源分配模式
        n_allocation_patterns % 资源分配模式数量
    end
    
    methods
        function obj = ImprovedTCSEnvironment(config)
            % 构造函数
            
            % 系统架构初始化
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            % 初始化组件重要性和映射
            obj.initializeComponents();
            
            % 攻击参数初始化
            obj.attack_types = config.attack_types;
            obj.attack_severity = config.attack_severity;
            obj.attack_detection_difficulty = config.attack_detection_difficulty;
            obj.n_attack_types = length(obj.attack_types);
            
            % 资源参数初始化
            obj.resource_types = config.resource_types;
            obj.resource_effectiveness = config.resource_effectiveness;
            obj.n_resource_types = length(obj.resource_types);
            obj.total_resources = config.total_resources;
            
            % 初始化简化的动作空间
            obj.initializeSimplifiedActions();
            
            % 计算状态和动作空间维度
            obj.calculateSpaceDimensions();
            
            % 初始化环境状态
            obj.reset();
        end
        
        function newObj = copy(obj)
            % 深拷贝方法
            newObj = ImprovedTCSEnvironment(struct());
            props = properties(obj);
            for i = 1:length(props)
                newObj.(props{i}) = obj.(props{i});
            end
            newObj.attack_history = [];
            newObj.defense_history = [];
            newObj.time_step = 0;
            newObj.current_state = obj.current_state;
        end
        
        function initializeComponents(obj)
            % 初始化组件重要性和站点映射
            obj.component_importance = zeros(1, obj.total_components);
            obj.component_station_map = zeros(1, obj.total_components);
            
            idx = 1;
            for s = 1:obj.n_stations
                for c = 1:obj.n_components(s)
                    % 组件重要性基于类型和位置
                    if c <= 2  % 核心组件
                        obj.component_importance(idx) = 0.8 + 0.2 * rand();
                    elseif c <= 4  % 重要组件
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
        
        function initializeSimplifiedActions(obj)
            % 初始化简化的资源分配模式
            obj.n_allocation_patterns = 10;  % 10种预定义的分配模式
            obj.simplified_actions = cell(1, obj.n_allocation_patterns);
            
            % 模式1：均匀分配
            obj.simplified_actions{1} = ones(obj.total_components, 1) / obj.total_components;
            
            % 模式2：重点保护核心组件
            obj.simplified_actions{2} = obj.component_importance' / sum(obj.component_importance);
            
            % 模式3-5：重点保护各主站
            for i = 3:min(obj.n_stations+2, 7)
                allocation = zeros(obj.total_components, 1);
                station_idx = i - 2;
                if station_idx <= obj.n_stations
                    components_in_station = find(obj.component_station_map == station_idx);
                    allocation(components_in_station) = 1 / length(components_in_station);
                    obj.simplified_actions{i} = allocation;
                end
            end
            
            % 模式6-10：混合策略
            for i = max(obj.n_stations+3, 8):obj.n_allocation_patterns
                % 随机组合策略
                weights = rand(2, 1);
                weights = weights / sum(weights);
                obj.simplified_actions{i} = weights(1) * obj.simplified_actions{1} + ...
                                           weights(2) * obj.simplified_actions{2};
            end
        end
        
        function calculateSpaceDimensions(obj)
            % 计算状态和动作空间维度（简化版）
            
            % 状态空间：简化到1000
            obj.state_dim = 1000;
            
            % 防御者动作：使用预定义的分配模式
            obj.action_dim_defender = obj.n_allocation_patterns;
            
            % 攻击者动作：攻击目标选择
            obj.action_dim_attacker = obj.total_components;
        end
        
        function state = reset(obj)
            % 重置环境
            obj.time_step = 0;
            obj.attack_history = [];
            obj.defense_history = [];
            obj.current_state = obj.generateInitialState();
            state = obj.current_state;
        end
        
        function state = generateInitialState(obj)
            % 生成初始状态（简化版）
            % 状态编码：最近的攻击模式 + 当前防御状态
            state = randi(obj.state_dim);
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_action, attacker_action)
            % 执行一步环境交互（优化版）
            
            % 验证动作合法性
            obj.validateActions(defender_action, attacker_action);
            
            % 获取防御资源分配（使用预定义模式）
            defense_allocation = obj.simplified_actions{defender_action};
            
            % 攻击目标
            attack_target = attacker_action;
            attack_type = randi(obj.n_attack_types);  % 随机攻击类型
            
            % 计算检测结果（优化的检测概率）
            [detected, detection_prob] = obj.calculateImprovedDetection(...
                defense_allocation, attack_target, attack_type);
            
            % 计算奖励（优化的奖励函数）
            [reward_def, reward_att] = obj.calculateImprovedRewards(...
                detected, attack_target, attack_type, defense_allocation);
            
            % 更新状态
            obj.updateState(defender_action, attack_target, attack_type, detected);
            next_state = obj.current_state;
            
            % 记录信息
            info.detected = detected;
            info.detection_prob = detection_prob;
            info.attack_target = attack_target;
            info.attack_type = attack_type;
            info.target_importance = obj.component_importance(attack_target);
            info.allocation_pattern = defender_action;
            
            obj.time_step = obj.time_step + 1;
        end
        
        function validateActions(obj, defender_action, attacker_action)
            % 验证动作合法性
            assert(defender_action >= 1 && defender_action <= obj.action_dim_defender, ...
                   sprintf('防御动作越界: %d (范围: 1-%d)', defender_action, obj.action_dim_defender));
            
            assert(attacker_action >= 1 && attacker_action <= obj.action_dim_attacker, ...
                   sprintf('攻击动作越界: %d (范围: 1-%d)', attacker_action, obj.action_dim_attacker));
        end
        
        function [detected, detection_prob] = calculateImprovedDetection(obj, defense_allocation, attack_target, attack_type)
            % 优化的检测概率计算
            
            % 目标组件的防御资源
            target_defense = defense_allocation(attack_target);
            
            % 基础检测概率（提高基础值）
            base_detection = 0.3;  % 基础30%检测率
            
            % 资源加成（线性关系，更容易学习）
            resource_bonus = target_defense * 0.6;  % 最多增加60%
            
            % 考虑攻击难度
            difficulty_factor = 1 - 0.5 * obj.attack_detection_difficulty(attack_type);
            
            % 最终检测概率
            detection_prob = (base_detection + resource_bonus) * difficulty_factor;
            detection_prob = max(0.1, min(0.95, detection_prob));
            
            % 判定检测
            detected = rand() < detection_prob;
        end
        
        function [reward_def, reward_att] = calculateImprovedRewards(obj, detected, attack_target, attack_type, defense_allocation)
            % 优化的奖励函数
            
            % 基础奖励
            target_importance = obj.component_importance(attack_target);
            attack_impact = target_importance * obj.attack_severity(attack_type);
            
            if detected
                % 检测成功的奖励（大幅提高）
                reward_def = 100;  % 固定高奖励
                reward_att = -50;
            else
                % 检测失败的惩罚
                reward_def = -50 * attack_impact;
                reward_att = 20;
            end
            
            % 额外奖励：鼓励关注重要组件
            if defense_allocation(attack_target) > 0.1 && target_importance > 0.7
                reward_def = reward_def + 10;  % 额外奖励
            end
            
            % 简化资源效率（不惩罚资源使用）
            % reward_def = reward_def - 0.001 * obj.total_resources;
        end
        
        function updateState(obj, defense_pattern, attack_target, attack_type, detected)
            % 更新环境状态（简化版）
            
            % 更新历史
            obj.attack_history(end+1, :) = [obj.time_step, attack_target, attack_type, detected];
            obj.defense_history(end+1, :) = [obj.time_step, defense_pattern];
            
            % 简单的状态转移
            % 基于最近的防御模式和攻击结果
            state_hash = defense_pattern * 100 + attack_target * 10 + detected * 5 + attack_type;
            obj.current_state = mod(state_hash, obj.state_dim) + 1;
        end
        
        function stats = getStatistics(obj)
            % 获取环境统计
            stats.total_attacks = size(obj.attack_history, 1);
            if stats.total_attacks > 0
                stats.detected_attacks = sum(obj.attack_history(:, 4));
                stats.detection_rate = stats.detected_attacks / stats.total_attacks;
            else
                stats.detected_attacks = 0;
                stats.detection_rate = 0;
            end
            
            % 攻击目标分布
            if stats.total_attacks > 0
                stats.target_distribution = zeros(1, obj.total_components);
                for i = 1:stats.total_attacks
                    target = obj.attack_history(i, 2);
                    stats.target_distribution(target) = stats.target_distribution(target) + 1;
                end
            end
        end
    end
end