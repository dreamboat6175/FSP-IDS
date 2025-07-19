% =========================================================================
% 修正Version 3.0中的逻辑问题，提供完整实现
% 1. 修正角色职责混乱
% 2. 修正step方法逻辑错误  
% 3. 修正FSP实现位置错误
% 4. 修正方法参数不匹配
% 5. 提供完整的代码实现
% =========================================================================

classdef TCSEnvironment < handle
    %TCSENVIRONMENT 列控系统网络安全对抗环境（完整修正版）
    
    properties (Access = public)
        %% === 基础系统参数 ===
        n_stations              % 站点数量
        n_components            % 每站点组件数量向量  
        total_components        % 总组件数
        station_values          % 站点价值权重向量 [0,1]
        total_resources         % 总防御资源数
        n_resource_types        % 资源类型数量
        n_attack_types          % 攻击类型数量
        
        %% === 环境状态 ===
        current_state          % 当前环境状态向量
        time_step             % 当前时间步
        
        %% === 状态和动作空间 ===
        state_dim             % 状态空间维度
        action_dim            % 动作空间维度
        action_dim_defender   % 防御者动作维度
        action_dim_attacker   % 攻击者动作维度
        
        %% === FSP相关（环境维护的共享信息） ===
        attacker_avg_strategy   % 攻击者历史平均策略
        alpha_ewma             % 策略平均更新参数
        
        %% === 内部Q-Learning状态（用于环境内部逻辑，非智能体） ===
        attacker_Q_table       % 环境维护的Q表（用于baseline对比）
        attacker_lr           % 内部Q学习参数
        attacker_gamma        
        attacker_epsilon      
        attacker_epsilon_decay
        attacker_epsilon_min
        max_defense_states    % Q表状态数限制
        
        %% === 性能评估参数 ===
        radi_score           % 当前RADI得分
        radi_config          % RADI配置参数
        
        %% === 检测系统参数 ===
        detection_enabled     % 检测系统启用状态
        base_detection_rate   % 基础检测率
        detection_sensitivity % 检测敏感度
        false_positive_rate   % 误报率
        
        %% === 历史记录 ===
        attack_success_history    % 攻击成功历史 [0,1]
        attack_target_history     % 攻击目标历史 [one-hot]
        defense_deployment_history % 防御部署历史
        damage_history            % 损害历史
        radi_history              % RADI历史
        detection_history         % 检测历史
        reward_history            % 奖励历史
        
        %% === 兼容性属性 ===
        defense_costs           % 防御成本
        defense_effectiveness   % 防御效果
        radi_defender          % 兼容性RADI
        radi_attacker          % 兼容性RADI
        defender_strategy      % 兼容性策略
        attacker_strategy      % 兼容性策略
        optimal_defender_strategy % 兼容性
        optimal_attacker_strategy % 兼容性
        deployment_history     % 兼容性历史
        damage_history_alt     % 兼容性历史
        attack_history         % 兼容性历史
        defense_history        % 兼容性历史
        attacker_actions       % 兼容性
        defender_actions       % 兼容性
        
        %% === 其他参数 ===
        epsilon               % 数值稳定性参数
        debug_mode           % 调试模式
        optimization_method  % 优化方法（兼容性）
    end
    
    properties (Access = private)
        %% === 私有辅助属性 ===
        action_templates     % 预定义动作模板
    end
    
    methods (Access = public)
        function obj = TCSEnvironment(config)
            %TCSENVIRONMENT 构造函数（修正版）
            
            if nargin < 1
                error('TCSEnvironment:InvalidInput', '需要配置参数config');
            end
            
            % 验证配置
            obj.validateConfig(config);
            
            % 初始化基础参数
            obj.initializeBasicParameters(config);
            
            % 初始化环境组件
            obj.initializeEnvironmentComponents(config);
            
            % 初始化历史记录
            obj.initializeHistoryRecords();
            
            % 计算空间维度
            obj.calculateSpaceDimensions();
            
            % 重置到初始状态
            obj.reset();
            
            if obj.debug_mode
                fprintf('[TCSEnvironment] 初始化完成 - %d站点, %d资源\n', ...
                        obj.n_stations, obj.total_resources);
            end
        end
        
        function state = reset(obj)
            %RESET 重置环境到初始状态
            
            obj.time_step = 0;
            
            % 重置FSP平均策略为均匀分布
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 重置内部Q表（如果使用）
            if ~isempty(obj.attacker_Q_table)
                obj.attacker_Q_table = randn(obj.max_defense_states, obj.n_stations) * 0.1;
            end
            
            % 重置兼容性策略
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.attacker_strategy = obj.station_values / sum(obj.station_values); % 基于价值初始化
            
            % 清空历史记录
            obj.clearHistoryRecords();
            
            % 生成初始状态
            obj.current_state = obj.generateEnvironmentState();
            state = obj.current_state;
            
            if obj.debug_mode
                fprintf('[TCSEnvironment] 环境已重置\n');
            end
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_deployment, attacker_target)
            
            % === 输入验证 ===
            obj.validateStepInputs(defender_deployment, attacker_target);
            
            % === 1. 更新FSP平均策略（环境维护的共享信息） ===
            obj.updateAttackerAverageStrategy(attacker_target);
            
            % === 2. 计算攻击结果 ===
            [attack_success, damage] = obj.computeAttackOutcome(attacker_target, defender_deployment);
            
            % === 3. 执行检测评估 ===
            detection_result = obj.evaluateDetection(attacker_target, defender_deployment, attack_success);
            
            % === 4. 计算奖励 ===
            [reward_def, reward_att] = obj.computeRewards(attack_success, damage, attacker_target, ...
                                                         defender_deployment, detection_result);
            
            % === 5. 更新环境状态 ===
            obj.updateEnvironmentState(attack_success, damage, attacker_target, ...
                                      defender_deployment, detection_result);
            
            % === 6. 生成下一状态 ===
            next_state = obj.generateEnvironmentState();
            obj.current_state = next_state;
            obj.time_step = obj.time_step + 1;
            
            % === 7. 准备返回信息 ===
           info = obj.createStepInfo(attack_success, damage, attacker_target, ...
                              defender_deployment, detection_result, ...
                              reward_def, reward_att);
    
            % 确保rewards结构正确
            rewards.defender = reward_def;
            rewards.attacker = reward_att;
        end
        
        function radi = calculateRADI(obj, current_allocation, optimal_allocation, varargin)
            %CALCULATERADI 计算RADI指标（修正版）
            
            % 处理输入参数
            if nargin < 3
                error('TCSEnvironment:InvalidInput', 'calculateRADI需要至少2个分配向量');
            end
            
            % 预处理分配向量
            [current_norm, optimal_norm] = obj.preprocessAllocations(current_allocation, optimal_allocation);
            
            if isempty(current_norm) || isempty(optimal_norm)
                radi = 0;
                return;
            end
            
            % 处理权重
            if nargin > 3 && ~isempty(varargin{1})
                weights = varargin{1};
            else
                weights = ones(1, length(current_norm)) / length(current_norm);
            end
            
            % 确保权重长度正确
            if length(weights) ~= length(current_norm)
                weights = weights(1:min(length(weights), length(current_norm)));
                if length(weights) < length(current_norm)
                    weights = [weights, ones(1, length(current_norm) - length(weights)) / length(current_norm)];
                end
            end
            weights = weights / sum(weights);
            
            % 计算RADI
            deviation = abs(current_norm - optimal_norm);
            radi = sum(weights .* deviation);
            
            % 数值稳定性检查
            if isnan(radi) || isinf(radi)
                radi = 0;
            end
        end
        
        function detection_rate = calculateDetectionRate(obj)
            %CALCULATEDETECTIONRATE 计算当前检测率
            
            if ~obj.detection_enabled
                detection_rate = 0;
                return;
            end
            
            % 基于最近防御历史计算
            if isempty(obj.defense_deployment_history)
                detection_rate = obj.base_detection_rate;
                return;
            end
            
            recent_window = min(50, size(obj.defense_deployment_history, 1));
            recent_deployments = obj.defense_deployment_history(end-recent_window+1:end, :);
            
            % 计算检测率
            avg_defense = mean(sum(recent_deployments, 2)) / obj.total_resources;
            detection_rate = obj.base_detection_rate + avg_defense * 0.4;
            detection_rate = max(0.1, min(0.9, detection_rate));
        end
        
        function state = generateEnvironmentState(obj)
            %GENERATEENVIRONMENTSTATE 生成环境状态向量
            
            % 状态组成：[攻击者平均策略, 时间归一化, 最近RADI]
            time_norm = min(obj.time_step / 1000, 1.0);
            recent_radi = obj.computeRecentRADI();
            
            state = [obj.attacker_avg_strategy, time_norm, recent_radi];
            
            % 确保状态长度正确
            if length(state) ~= obj.state_dim
                if length(state) < obj.state_dim
                    state = [state, zeros(1, obj.state_dim - length(state))];
                else
                    state = state(1:obj.state_dim);
                end
            end
        end
        
        function info = getEnvironmentInfo(obj)
            %GETENVIRONMENTINFO 获取环境信息
            
            info = struct();
            info.n_stations = obj.n_stations;
            info.total_resources = obj.total_resources;
            info.time_step = obj.time_step;
            info.current_radi = obj.radi_score;
            
            % 性能统计
            if ~isempty(obj.attack_success_history)
                recent_window = min(50, length(obj.attack_success_history));
                info.recent_success_rate = mean(obj.attack_success_history(end-recent_window+1:end));
            else
                info.recent_success_rate = 0;
            end
            
            if ~isempty(obj.detection_history)
                recent_window = min(50, length(obj.detection_history));
                info.recent_detection_rate = mean(obj.detection_history(end-recent_window+1:end));
            else
                info.recent_detection_rate = 0;
            end
            
            info.recent_radi = obj.computeRecentRADI();
        end
        
        %% === 兼容性方法 ===
        
        function deployment = parseDefenderAction(obj, action)
            %PARSEDEFENDERACTION 解析防御者动作（兼容性）
            
            if isscalar(action)
                % 使用预定义模板
                template_idx = min(max(round(action), 1), size(obj.action_templates, 1));
                deployment = obj.action_templates(template_idx, :) * obj.total_resources;
            elseif length(action) == obj.n_stations
                deployment = action;
                if sum(deployment) > 0
                    deployment = deployment / sum(deployment) * obj.total_resources;
                end
            else
                deployment = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
            end
            
            deployment = max(0, deployment);
        end
        
        function target = parseAttackerAction(obj, action)
            %PARSEATTACKERACTION 解析攻击者动作（兼容性）
            
            if isscalar(action)
                target = min(max(round(action), 1), obj.n_stations);
            else
                [~, target] = max(action);
            end
        end
        
        function [reward_def, reward_att] = calculateRewards(obj, attack_success, damage, attacker_target, defender_deployment)
            %CALCULATEREWARDS 计算奖励（兼容性别名）
            
            detection_result = struct('detected', false, 'detection_prob', 0, 'is_false_positive', false);
            [reward_def, reward_att] = obj.computeRewards(attack_success, damage, attacker_target, ...
                                                         defender_deployment, detection_result);
        end 
    end
    
    methods (Access = private)
        function validateConfig(obj, config)
            %VALIDATECONFIG 验证配置参数
            
            if ~isfield(config, 'n_stations') || config.n_stations <= 0
                error('TCSEnvironment:InvalidConfig', '需要有效的n_stations参数');
            end
        end
        
        function initializeBasicParameters(obj, config)
            %INITIALIZEBASICPARAMETERS 初始化基础参数
            
            % 基础系统参数
            obj.n_stations = config.n_stations;
            
            % 组件配置
            if isfield(config, 'n_components_per_station') && ~isempty(config.n_components_per_station)
                obj.n_components = config.n_components_per_station(:)';
                if length(obj.n_components) ~= obj.n_stations
                    error('TCSEnvironment:InvalidConfig', 'n_components_per_station长度必须等于n_stations');
                end
            else
                obj.n_components = ones(1, obj.n_stations) * 3;
            end
            obj.total_components = sum(obj.n_components);
            
            % 资源参数
            obj.total_resources = obj.getConfigValue(config, 'total_resources', 100);
            obj.n_resource_types = obj.getConfigValue(config, 'n_resource_types', 5);
            obj.n_attack_types = obj.getConfigValue(config, 'n_attack_types', 6);
            
            % 其他参数
            obj.epsilon = 1e-6;
            obj.debug_mode = obj.getConfigValue(config, 'debug_mode', false);
            obj.optimization_method = 'analytical';
            
            % 生成站点价值
            obj.generateStationValues();
        end
        
        function initializeEnvironmentComponents(obj, config)
            %INITIALIZEENVIRONMENTCOMPONENTS 初始化环境组件
            
            % FSP参数
            obj.alpha_ewma = obj.getConfigValue(config, 'alpha_ewma', 0.1);
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 内部Q-Learning参数（用于baseline）
            obj.attacker_lr = obj.getConfigValue(config, 'attacker_lr', 0.1);
            obj.attacker_gamma = obj.getConfigValue(config, 'attacker_gamma', 0.95);
            obj.attacker_epsilon = obj.getConfigValue(config, 'attacker_epsilon', 0.3);
            obj.attacker_epsilon_decay = obj.getConfigValue(config, 'attacker_epsilon_decay', 0.995);
            obj.attacker_epsilon_min = obj.getConfigValue(config, 'attacker_epsilon_min', 0.01);
            obj.max_defense_states = 50;
            obj.attacker_Q_table = randn(obj.max_defense_states, obj.n_stations) * 0.1;
            
            % RADI配置
            obj.radi_config = struct();
            if isfield(config, 'radi')
                obj.radi_config = config.radi;
            else
                obj.radi_config.optimal_allocation = ones(1, obj.n_stations) / obj.n_stations;
                obj.radi_config.weights = ones(1, obj.n_stations) / obj.n_stations;
            end
            
            % 检测系统
            obj.detection_enabled = obj.getConfigValue(config, 'detection_enabled', true);
            obj.base_detection_rate = obj.getConfigValue(config, 'base_detection_rate', 0.3);
            obj.detection_sensitivity = obj.getConfigValue(config, 'detection_sensitivity', 0.8);
            obj.false_positive_rate = obj.getConfigValue(config, 'false_positive_rate', 0.1);
            
            % 创建动作模板
            obj.action_templates = obj.createActionTemplates();
            
            % 兼容性初始化
            obj.initializeCompatibilityAttributes();
        end
        
        function initializeCompatibilityAttributes(obj)
            %INITIALIZECOMPATIBILITYATTRIBUTES 初始化兼容性属性
            
            obj.defense_costs = ones(1, obj.n_stations);
            obj.defense_effectiveness = ones(1, obj.n_stations) * 0.8;
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.attacker_strategy = obj.station_values / sum(obj.station_values);
            obj.optimal_defender_strategy = obj.defender_strategy;
            obj.optimal_attacker_strategy = obj.attacker_strategy;
            obj.attacker_actions = [];
            obj.defender_actions = [];
        end
        
        function initializeHistoryRecords(obj)
            %INITIALIZEHISTORYRECORDS 初始化历史记录
            
            obj.attack_success_history = [];
            obj.attack_target_history = [];
            obj.defense_deployment_history = [];
            obj.damage_history = [];
            obj.radi_history = [];
            obj.detection_history = [];
            obj.reward_history = struct('defender', [], 'attacker', []);
            
            % 兼容性历史
            obj.deployment_history = [];
            obj.damage_history_alt = [];
            obj.attack_history = [];
            obj.defense_history = [];
        end
        
        function clearHistoryRecords(obj)
            %CLEARHISTORYRECORDS 清空历史记录
            
            obj.attack_success_history = [];
            obj.attack_target_history = [];
            obj.defense_deployment_history = [];
            obj.damage_history = [];
            obj.radi_history = [];
            obj.detection_history = [];
            obj.reward_history = struct('defender', [], 'attacker', []);
            
            % 清空兼容性历史
            obj.deployment_history = [];
            obj.damage_history_alt = [];
            obj.attack_history = [];
            obj.defense_history = [];
        end
        
        function calculateSpaceDimensions(obj)
            %CALCULATESPACEDIMENSIONS 计算状态和动作空间维度
            
            % 状态维度：[攻击者平均策略, 时间归一化, RADI]
            obj.state_dim = obj.n_stations + 1 + 1;
            
            % 动作维度
            obj.action_dim = obj.n_stations;
            obj.action_dim_defender = obj.n_stations;
            obj.action_dim_attacker = obj.n_stations;
        end
        
        function generateStationValues(obj)
            %GENERATESTATIONVALUES 生成站点价值
            
            % 基于组件数量生成价值
            component_importance = rand(1, obj.total_components);
            obj.station_values = zeros(1, obj.n_stations);
            
            idx = 1;
            for i = 1:obj.n_stations
                n_comp = obj.n_components(i);
                obj.station_values(i) = sum(component_importance(idx:idx+n_comp-1));
                idx = idx + n_comp;
            end
            
            % 归一化并增加差异性
            obj.station_values = obj.station_values / sum(obj.station_values);
            obj.station_values = obj.station_values .^ 0.8; % 减少极端差异
            obj.station_values = obj.station_values / sum(obj.station_values);
        end
        
        function validateStepInputs(obj, defender_deployment, attacker_target)
            %VALIDATESTEPINPUTS 验证step方法输入
            
            if length(defender_deployment) ~= obj.n_stations
                error('TCSEnvironment:InvalidInput', '防御部署向量长度必须为%d', obj.n_stations);
            end
            
            if ~isscalar(attacker_target) || attacker_target < 1 || attacker_target > obj.n_stations
                error('TCSEnvironment:InvalidInput', '攻击目标必须在1到%d之间', obj.n_stations);
            end
            
            if any(defender_deployment < 0)
                error('TCSEnvironment:InvalidInput', '防御部署不能为负值');
            end
        end
        
        function updateAttackerAverageStrategy(obj, attacker_target)
            %UPDATEATTACKERAVERAGESTRATEGY 更新攻击者平均策略
            
            % 创建当前动作的one-hot向量
            current_action = zeros(1, obj.n_stations);
            current_action(attacker_target) = 1;
            
            % EWMA更新
            obj.attacker_avg_strategy = (1 - obj.alpha_ewma) * obj.attacker_avg_strategy + ...
                                       obj.alpha_ewma * current_action;
            
            % 归一化
            obj.attacker_avg_strategy = obj.attacker_avg_strategy / sum(obj.attacker_avg_strategy);
        end
        
        function [attack_success, damage] = computeAttackOutcome(obj, attacker_target, defender_deployment)
            %COMPUTEATTACKOUTCOME 计算攻击结果（改进版）
            
            % 目标站点防御强度
            defense_strength = defender_deployment(attacker_target);
            defense_ratio = defense_strength / obj.total_resources;
            
            % 改进的攻击成功概率模型
            base_success_prob = 0.6; % 基础成功率
            defense_effect = 1 / (1 + exp(8 * (defense_ratio - 0.4))); % sigmoid函数
            success_prob = base_success_prob * defense_effect;
            
            % 添加随机噪声
            noise = (rand() - 0.5) * 0.1;
            success_prob = max(0.05, min(0.95, success_prob + noise));
            
            % 确定攻击结果
            attack_success = rand() < success_prob;
            
            % 计算损害
            if attack_success
                base_damage = obj.station_values(attacker_target);
                damage_reduction = defense_ratio * 0.6; % 防御可减少60%损害
                damage = base_damage * (1 - damage_reduction);
            else
                damage = 0;
            end
            
            % 调试输出
            if obj.debug_mode && mod(obj.time_step, 50) == 0
                fprintf('[Debug] 目标%d: 防御=%.2f, 成功率=%.2f, 结果=%d\n', ...
                        attacker_target, defense_ratio, success_prob, attack_success);
            end
        end
        
        function detection_result = evaluateDetection(obj, attacker_target, defender_deployment, attack_success)
            %EVALUATEDETECTION 评估检测结果
            
            detection_result = struct();
            
            if ~obj.detection_enabled
                detection_result.detected = false;
                detection_result.detection_prob = 0;
                detection_result.is_false_positive = false;
                return;
            end
            
            % 计算检测概率
            base_prob = obj.detection_sensitivity;
            defense_boost = (defender_deployment(attacker_target) / obj.total_resources) * 0.3;
            success_boost = attack_success * 0.2;
            
            detection_prob = base_prob + defense_boost + success_boost;
            detection_prob = max(0.05, min(0.95, detection_prob));
            
            % 确定检测结果
            detected = rand() < detection_prob;
            
            % 处理误报
            is_false_positive = false;
            if ~attack_success && rand() < obj.false_positive_rate
                detected = true;
                is_false_positive = true;
            end
            
            detection_result.detected = detected;
            detection_result.detection_prob = detection_prob;
            detection_result.is_false_positive = is_false_positive;
        end
        
        function [reward_def, reward_att] = computeRewards(obj, attack_success, damage, ...
    attacker_target, defender_deployment, detection_result)
    % 修复的奖励函数 - 确保防御者获得正向激励
    
    %% === 攻击者奖励 ===
    if attack_success
        base_reward = obj.station_values(attacker_target) * 2;
        defense_penalty = -(defender_deployment(attacker_target) / obj.total_resources) * 0.5;
        if detection_result.detected
            detection_penalty = -0.5;
        else
            detection_penalty = 0.1;
        end
        reward_att = base_reward + defense_penalty + detection_penalty;
    else
        reward_att = -0.1;
    end
    
    %% === 防御者奖励（修复版）===
    
    % 1. 防御成功/失败奖励（增加正向激励）
    if attack_success
        defense_reward = -0.5;  % 减少惩罚
    else
        defense_reward = 2.0;   % 增加奖励
    end
    
    % 2. RADI性能奖励（确保为正值）
    optimal_deployment = obj.computeOptimalDeployment(attacker_target);
    current_radi = obj.calculateRADI(defender_deployment, optimal_deployment);
    obj.radi_score = current_radi;
    
    % 使用指数函数确保正值
    radi_reward = exp(-current_radi * 2);  % 范围[0, 1]
    
    % 3. 检测系统奖励
    detection_reward = 0;
    if detection_result.detected && ~detection_result.is_false_positive
        detection_reward = 0.3;
    elseif detection_result.is_false_positive
        detection_reward = -0.05;  % 减少误报惩罚
    end
    
    % 4. 损害奖励
    damage_reward = (1 - damage) * 0.5;
    
    % 5. 资源效率奖励
    total_used = sum(defender_deployment);
    resource_efficiency = 1 - abs(total_used - obj.total_resources) / obj.total_resources;
    efficiency_reward = max(0, resource_efficiency * 0.2);
    
    % 综合奖励计算（添加基础奖励确保正值）
    reward_def = 1.0 + ...  % 基础奖励
                 defense_reward * 0.3 + ...
                 radi_reward * 0.3 + ...
                 detection_reward * 0.2 + ...
                 damage_reward * 0.1 + ...
                 efficiency_reward * 0.1;
    
    % 确保奖励在合理范围内
    reward_def = max(-1, min(5, reward_def));
end
        
        function optimal = computeOptimalDeployment(obj, attacker_target)
            %COMPUTEOPTIMALDEPLOYMENT 计算事后最优部署
            
            optimal = zeros(1, obj.n_stations);
            
            % 40%资源给被攻击站点
            main_alloc = 0.6;
            optimal(attacker_target) = obj.total_resources * main_alloc;
            
            % 剩余60%按价值和威胁分配
            remaining = obj.total_resources * (1 - main_alloc);
            other_stations = setdiff(1:obj.n_stations, attacker_target);
            
            if ~isempty(other_stations)
                % 结合站点价值和感知威胁
                values = obj.station_values(other_stations);
                threats = obj.attacker_avg_strategy(other_stations);
                combined_weights = values .* (1 + threats); % 价值×(1+威胁)
                
                if sum(combined_weights) > 0
                    combined_weights = combined_weights / sum(combined_weights);
                    for i = 1:length(other_stations)
                        optimal(other_stations(i)) = remaining * combined_weights(i);
                    end
                else
                    % 均匀分配剩余资源
                    warning('Deployment:Fallback', '组合权重为零，无法进行加权分配。正在对剩余资源执行均匀分配。');
                    for i = 1:length(other_stations)
                        optimal(other_stations(i)) = remaining / length(other_stations);
                    end
                end
            end
        end
        
        function updateEnvironmentState(obj, attack_success, damage, attacker_target, defender_deployment, detection_result)
            %UPDATEENVIRONMENTSTATE 更新环境状态
            
            % 记录基本历史
            obj.attack_success_history(end+1) = double(attack_success);
            obj.damage_history(end+1) = damage;
            obj.radi_history(end+1) = obj.radi_score;
            obj.detection_history(end+1) = detection_result.detected;
            
            % 记录攻击目标（one-hot编码）
            target_vector = zeros(1, obj.n_stations);
            target_vector(attacker_target) = 1;
            
            if isempty(obj.attack_target_history)
                obj.attack_target_history = target_vector;
                obj.defense_deployment_history = defender_deployment;
            else
                obj.attack_target_history(end+1, :) = target_vector;
                obj.defense_deployment_history(end+1, :) = defender_deployment;
            end
            
            % 更新兼容性历史
            obj.deployment_history = obj.defense_deployment_history;
            obj.damage_history_alt = obj.damage_history;
            obj.attack_history(end+1) = attacker_target;
            obj.defense_history = obj.defense_deployment_history;
            
            % 更新兼容性策略和RADI
            obj.radi_defender = obj.radi_score;
            obj.radi_attacker = 0;
            
            % 更新当前策略（兼容性）
            if sum(defender_deployment) > 0
                obj.defender_strategy = defender_deployment / sum(defender_deployment);
            end
            
            % 更新攻击者策略（基于最近行为）
            recent_window = min(10, length(obj.attack_history));
            if recent_window > 0
                recent_targets = obj.attack_history(end-recent_window+1:end);
                strategy_update = zeros(1, obj.n_stations);
                for target = recent_targets
                    strategy_update(target) = strategy_update(target) + 1;
                end
                if sum(strategy_update) > 0
                    obj.attacker_strategy = strategy_update / sum(strategy_update);
                end
            end
        end
        
        function info = createStepInfo(obj, attack_success, damage, attacker_target, ...
    defender_deployment, detection_result, reward_def, reward_att)
    % 创建步骤信息结构体（确保包含所有必要字段）
    
    info = struct();
    
    % 基本信息
    info.attack_success = attack_success;
    info.damage = damage;
    info.attacker_target = attacker_target;
    info.defender_deployment = defender_deployment;
    info.time_step = obj.time_step;
    
    % 性能指标
    info.radi_score = obj.radi_score;
    info.detection_result = detection_result;  % 确保包含完整的检测结果
    
    % 奖励信息
    info.reward_def = reward_def;
    info.reward_att = reward_att;
    
    % 统计信息
    info.recent_success_rate = obj.computeRecentSuccessRate();
    info.recent_detection_rate = obj.computeRecentDetectionRate();
    info.recent_radi = obj.computeRecentRADI();
    
    % 资源分配信息（用于性能监控）
    info.resource_allocation = defender_deployment / sum(defender_deployment);
end
        
        function [current_norm, optimal_norm] = preprocessAllocations(obj, current, optimal)
            %PREPROCESSALLOCATIONS 预处理分配向量
            
            % 类型转换和验证
            if iscell(current), current = cell2mat(current); end
            if iscell(optimal), optimal = cell2mat(optimal); end
            
            current = double(current);
            optimal = double(optimal);
            
            % 长度对齐
            n = min(length(current), length(optimal));
            if n == 0
                current_norm = [];
                optimal_norm = [];
                return;
            end
            
            current = current(1:n);
            optimal = optimal(1:n);
            
            % 归一化处理
            if sum(current) > obj.epsilon
                current_norm = current / sum(current);
            else
                current_norm = ones(1, n) / n;
            end
            
            if sum(optimal) > obj.epsilon
                optimal_norm = optimal / sum(optimal);
            else
                optimal_norm = ones(1, n) / n;
            end
        end
        
        function templates = createActionTemplates(obj)
            %CREATEACTIONTEMPLATES 创建预定义动作模板
            
            n_templates = 6;
            templates = zeros(n_templates, obj.n_stations);
            
            % 模板1: 均匀分配
            templates(1, :) = ones(1, obj.n_stations) / obj.n_stations;
            
            % 模板2: 基于价值分配
            templates(2, :) = obj.station_values / sum(obj.station_values);
            
            % 模板3: 集中防御最重要的站点
            [~, top_indices] = sort(obj.station_values, 'descend');
            n_focus = min(2, obj.n_stations);
            templates(3, top_indices(1:n_focus)) = 1 / n_focus;
            
            % 模板4: 集中防御前三个站点
            if obj.n_stations >= 3
                templates(4, top_indices(1:3)) = 1/3;
            else
                templates(4, :) = templates(1, :);
            end
            
            % 模板5: 随机分配（固定种子保证一致性）
            rng(42); % 固定随机种子
            random_weights = rand(1, obj.n_stations);
            templates(5, :) = random_weights / sum(random_weights);
            rng('shuffle'); % 恢复随机种子
            
            % 模板6: 反价值分配（保护价值较低的站点）
            inv_values = 1 ./ (obj.station_values + obj.epsilon);
            templates(6, :) = inv_values / sum(inv_values);
            
            % 确保所有模板归一化
            for i = 1:n_templates
                if sum(templates(i, :)) > 0
                    templates(i, :) = templates(i, :) / sum(templates(i, :));
                else
                    templates(i, :) = ones(1, obj.n_stations) / obj.n_stations;
                end
            end
        end
        
        function success_rate = computeRecentSuccessRate(obj)
            %COMPUTERECENTISUCCESSRATE 计算最近攻击成功率
            
            if isempty(obj.attack_success_history)
                success_rate = 0;
            else
                window = min(50, length(obj.attack_success_history));
                success_rate = mean(obj.attack_success_history(end-window+1:end));
            end
        end
        
        function detection_rate = computeRecentDetectionRate(obj)
    % 修复：正确计算最近检测率
    
    if isempty(obj.detection_history) || isempty(obj.attack_success_history)
        detection_rate = 0;
        return;
    end
    
    % 获取最近的窗口
    window = min(50, length(obj.attack_success_history));
    recent_attacks = obj.attack_success_history(end-window+1:end);
    recent_detections = obj.detection_history(end-window+1:end);
    
    % 找出成功的攻击
    success_indices = find(recent_attacks == 1);
    
    if isempty(success_indices)
        % 没有成功攻击，检测率无意义
        detection_rate = 0;
    else
        % 计算成功攻击中被检测到的比例
        detected_count = sum(recent_detections(success_indices));
        detection_rate = detected_count / length(success_indices);
    end
end
        
        function radi_mean = computeRecentRADI(obj)
            %COMPUTERECENTIRÁDI 计算最近RADI均值
            
            if isempty(obj.radi_history)
                radi_mean = 0;
            else
                window = min(50, length(obj.radi_history));
                radi_mean = mean(obj.radi_history(end-window+1:end));
            end
        end
        
        function state_idx = encodeDefenderState(obj, deployment)
            %ENCODEDEFENDERSTATE 编码防御状态为离散索引
            
            % 归一化部署
            if sum(deployment) > obj.epsilon
                normalized = deployment / sum(deployment);
            else
                normalized = ones(1, obj.n_stations) / obj.n_stations;
            end
            
            % 简化编码：主防御站点 + 资源集中度
            [~, primary_station] = max(normalized);
            concentration = max(normalized);
            
            % 离散化集中度为5个级别
            concentration_level = min(floor(concentration * 5), 4);
            
            % 计算状态索引
            state_idx = (primary_station - 1) * 5 + concentration_level + 1;
            state_idx = min(max(state_idx, 1), obj.max_defense_states);
        end
        
        function choice = weightedRandomChoice(obj, weights)
            %WEIGHTEDRANDOMCHOICE 基于权重的随机选择
            
            % 归一化权重
            weights = weights / sum(weights);
            
            % 累积分布
            cumulative = cumsum(weights);
            
            % 随机选择
            r = rand();
            choice = find(r <= cumulative, 1, 'first');
            
            if isempty(choice)
                choice = length(weights);
            end
        end
        
        function value = getConfigValue(obj, config, field, default_value)
            %GETCONFIGVALUE 安全获取配置值
            
            if isfield(config, field) && ~isempty(config.(field))
                value = config.(field);
            else
                value = default_value;
            end
        end
    end
    
    methods (Static)
        function config = getDefaultConfig()
            %GETDEFAULTCONFIG 获取默认配置
            
            config = struct();
            
            % === 基础系统参数 ===
            config.n_stations = 5;
            config.n_components_per_station = [3, 3, 3, 3, 3];
            config.total_resources = 100;
            config.n_resource_types = 5;
            config.n_attack_types = 6;
            
            % === FSP参数 ===
            config.alpha_ewma = 0.1;
            
            % === 内部Q-Learning参数（用于baseline） ===
            config.attacker_lr = 0.1;
            config.attacker_gamma = 0.95;
            config.attacker_epsilon = 0.3;
            config.attacker_epsilon_decay = 0.995;
            config.attacker_epsilon_min = 0.01;
            
            % === RADI配置 ===
            config.radi = struct();
            config.radi.optimal_allocation = ones(1, 5) / 5;
            config.radi.weights = ones(1, 5) / 5;
            
            % === 检测系统参数 ===
            config.detection_enabled = true;
            config.base_detection_rate = 0.3;
            config.detection_sensitivity = 0.8;
            config.false_positive_rate = 0.1;
            
            % === 其他参数 ===
            config.debug_mode = false;
        end
        
        function config = getOptimizedConfig()
            %GETOPTIMIZEDCONFIG 获取优化配置（针对性能问题）
            
            config = TCSEnvironment.getDefaultConfig();
            
            % === 性能优化参数 ===
            config.alpha_ewma = 0.05;  % 更慢的策略更新
            config.attacker_lr = 0.03; % 更低的学习率
            config.attacker_epsilon = 0.5; % 更高的初始探索率
            config.attacker_epsilon_decay = 0.9995; % 更慢的衰减
            config.attacker_epsilon_min = 0.1; % 更高的最小探索率
            
            % === 检测系统增强 ===
            config.base_detection_rate = 0.4; % 更高的基础检测率
            config.detection_sensitivity = 0.85; % 更高的检测敏感度
            config.false_positive_rate = 0.08; % 略降低误报率
            
            % === 调试和监控 ===
            config.debug_mode = true;
        end
        
        function config = getTestConfig()
            %GETTESTCONFIG 获取测试配置（小规模快速测试）
            
            config = TCSEnvironment.getDefaultConfig();
            
            % 小规模测试参数
            config.n_stations = 3;
            config.n_components_per_station = [2, 2, 2];
            config.total_resources = 50;
            config.debug_mode = true;
            
            % 快速收敛参数
            config.alpha_ewma = 0.2;
            config.attacker_lr = 0.2;
            config.attacker_epsilon_decay = 0.99;
        end
        
        function demo()
            %DEMO 演示TCSEnvironment的使用（完整版）
            
            fprintf('=== TCSEnvironment v3.1 完整演示 ===\n\n');
            
            % === 1. 基础功能演示 ===
            fprintf('1. 基础功能演示\n');
            config = TCSEnvironment.getOptimizedConfig();
            config.debug_mode = false; % 关闭调试以减少输出
            env = TCSEnvironment(config);
            
            fprintf('   环境创建: %d站点, %d总资源\n', env.n_stations, env.total_resources);
            
            % === 2. 环境重置演示 ===
            fprintf('2. 环境重置演示\n');
            initial_state = env.reset();
            fprintf('   初始状态维度: %d\n', length(initial_state));
            fprintf('   站点价值: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.station_values);
            
            % === 3. 单步交互演示 ===
            fprintf('3. 单步交互演示\n');
            
            % 使用不同的防御策略
            strategies = {
                '均匀分配', ones(1, env.n_stations) * (env.total_resources / env.n_stations);
                '基于价值', env.station_values / sum(env.station_values) * env.total_resources;
                '集中防御', [env.total_resources*0.6, env.total_resources*0.1, env.total_resources*0.1, env.total_resources*0.1, env.total_resources*0.1]
            };
            
            for i = 1:size(strategies, 1)
                strategy_name = strategies{i, 1};
                defender_deployment = strategies{i, 2};
                attacker_target = randi(env.n_stations); % 随机攻击目标
                
                [next_state, reward_def, reward_att, info] = env.step(defender_deployment, attacker_target);
                
                fprintf('   策略: %s\n', strategy_name);
                fprintf('     攻击目标: %d, 成功: %d, 检测: %d\n', ...
                        attacker_target, info.attack_success, info.detection_result.detected);
                fprintf('     奖励: 防御者=%.2f, 攻击者=%.2f, RADI=%.4f\n', ...
                        reward_def, reward_att, info.radi_score);
                fprintf('     资源利用率: %.1f%%\n', info.resource_utilization * 100);
            end
            
            % === 4. 多步交互演示 ===
            fprintf('4. 多步交互演示（模拟训练）\n');
            env.reset();
            
            success_rates = [];
            detection_rates = [];
            radi_scores = [];
            
            for step = 1:20
                % 模拟智能防御策略（基于威胁感知）
                threat_weights = env.attacker_avg_strategy .* env.station_values;
                if sum(threat_weights) > 0
                    defense_weights = threat_weights / sum(threat_weights);
                else
                    defense_weights = ones(1, env.n_stations) / env.n_stations;
                end
                defender_deployment = defense_weights * env.total_resources;
                
                % 模拟智能攻击策略（基于价值和防御薄弱性）
                defense_strength = defender_deployment / env.total_resources;
                attack_attractiveness = env.station_values ./ (defense_strength + 0.1);
                [~, attacker_target] = max(attack_attractiveness);
                
                % 执行交互
                [~, ~, ~, info] = env.step(defender_deployment, attacker_target);
                
                % 记录性能
                success_rates(end+1) = info.recent_success_rate;
                detection_rates(end+1) = info.recent_detection_rate;
                radi_scores(end+1) = info.radi_score;
            end
            
            fprintf('   20步后性能统计:\n');
            fprintf('     平均攻击成功率: %.1f%%\n', mean(success_rates) * 100);
            fprintf('     平均检测率: %.1f%%\n', mean(detection_rates) * 100);
            fprintf('     最终RADI: %.4f\n', radi_scores(end));
            
            % === 5. 兼容性方法演示 ===
            fprintf('5. 兼容性方法演示\n');
            
            % 测试动作解析
            parsed_defense = env.parseDefenderAction(1); % 使用模板1
            parsed_attack = env.parseAttackerAction(2);  % 攻击站点2
            
            fprintf('   动作解析:\n');
            fprintf('     防御模板1: [%.1f, %.1f, %.1f, %.1f, %.1f]\n', parsed_defense);
            fprintf('     攻击目标: %d\n', parsed_attack);
            
            % 测试RADI计算
            current = [0.3, 0.2, 0.2, 0.15, 0.15];
            optimal = [0.2, 0.2, 0.2, 0.2, 0.2];
            radi = env.calculateRADI(current, optimal);
            
            fprintf('   RADI计算: %.4f\n', radi);
            
            % 测试检测率计算
            detection_rate = env.calculateDetectionRate();
            fprintf('   当前检测率: %.1f%%\n', detection_rate * 100);
            
            % === 6. 环境信息获取 ===
            fprintf('6. 环境信息获取\n');
            env_info = env.getEnvironmentInfo();
            
            fprintf('   环境统计:\n');
            fprintf('     时间步: %d\n', env_info.time_step);
            fprintf('     成功率: %.1f%%\n', env_info.recent_success_rate * 100);
            fprintf('     检测率: %.1f%%\n', env_info.recent_detection_rate * 100);
            fprintf('     RADI: %.4f\n', env_info.recent_radi);
            
            fprintf('\n=== 演示完成 ===\n');
        end
        
        function runTest()
            %RUNTEST 运行完整功能测试
            
            fprintf('=== TCSEnvironment v3.1 功能测试 ===\n\n');
            
            try
                % === 测试7: 边界条件处理 ===
                fprintf('测试7: 边界条件处理\n');
                
                % 测试异常输入处理
                try
                    env.step([1, 2], 1); % 错误的部署向量长度
                    assert(false, '应该抛出长度错误');
                catch ME
                    assert(contains(ME.message, '长度'), '错误信息不正确');
                end
                
                try
                    env.step([10, 20, 20], 0); % 错误的目标范围
                    assert(false, '应该抛出范围错误');
                catch ME
                    assert(contains(ME.message, '之间'), '错误信息不正确');
                end
                
                % 测试正常边界值
                [~, ~, ~, ~] = env.step([0, 0, 50], 3); % 极端部署
                [~, ~, ~, ~] = env.step([50, 0, 0], 1); % 集中部署
                
                fprintf('   ✓ 边界条件处理测试通过\n');
                
                % === 测试8: 性能指标一致性 ===
                fprintf('测试8: 性能指标一致性\n');
                
                env.reset();
                
                % 执行多步，验证统计一致性
                success_count = 0;
                for step = 1:20
                    [~, ~, ~, info] = env.step([15, 15, 20], randi(3));
                    if info.attack_success
                        success_count = success_count + 1;
                    end
                end
                
                manual_success_rate = success_count / 20;
                reported_success_rate = info.recent_success_rate;
                
                % 允许小的数值误差
                assert(abs(manual_success_rate - reported_success_rate) < 0.1, ...
                       '成功率统计不一致');
                
                fprintf('   ✓ 性能指标一致性测试通过\n');
                
                fprintf('\n=== 所有测试通过 ===\n');
                
            catch ME
                fprintf('\n❌ 测试失败: %s\n', ME.message);
                fprintf('   位置: %s\n', ME.stack(1).name);
                rethrow(ME);
            end
        end
        
        function runPerformanceTest()
            %RUNPERFORMANCETEST 运行性能测试
            
            fprintf('=== TCSEnvironment v3.1 性能测试 ===\n\n');
            
            % 创建测试配置
            config = TCSEnvironment.getOptimizedConfig();
            config.debug_mode = false;
            
            fprintf('测试配置: %d站点, %d资源\n', config.n_stations, config.total_resources);
            
            % 性能测试参数
            n_episodes = 50;
            n_steps_per_episode = 100;
            
            fprintf('开始性能测试: %d episodes × %d steps\n\n', n_episodes, n_steps_per_episode);
            
            % 创建环境
            env = TCSEnvironment(config);
            
            % 性能记录
            all_success_rates = [];
            all_detection_rates = [];
            all_radi_scores = [];
            episode_times = [];
            
            total_start = tic;
            
            for episode = 1:n_episodes
                episode_start = tic;
                env.reset();
                
                episode_success = [];
                episode_detection = [];
                episode_radi = [];
                
                for step = 1:n_steps_per_episode
                    % 使用智能策略模拟真实训练
                    % 防御策略：基于威胁感知
                    if step == 1
                        defender_deployment = ones(1, env.n_stations) * (env.total_resources / env.n_stations);
                    else
                        threat_weights = env.attacker_avg_strategy .* env.station_values;
                        if sum(threat_weights) > 0
                            defense_weights = threat_weights / sum(threat_weights);
                        else
                            defense_weights = ones(1, env.n_stations) / env.n_stations;
                        end
                        defender_deployment = defense_weights * env.total_resources;
                    end
                    
                    % 攻击策略：基于价值和防御薄弱性
                    defense_strength = defender_deployment / env.total_resources;
                    attack_attractiveness = env.station_values ./ (defense_strength + 0.1);
                    [~, attacker_target] = max(attack_attractiveness);
                    
                    % 执行交互
                    [~, ~, ~, info] = env.step(defender_deployment, attacker_target);
                    
                    % 记录性能
                    episode_success(end+1) = info.attack_success;
                    episode_detection(end+1) = info.detection_result.detected;
                    episode_radi(end+1) = info.radi_score;
                end
                
                episode_time = toc(episode_start);
                episode_times(end+1) = episode_time;
                
                % 记录episode统计
                all_success_rates(end+1) = mean(episode_success);
                all_detection_rates(end+1) = mean(episode_detection);
                all_radi_scores(end+1) = mean(episode_radi);
                
                % 进度报告
                if mod(episode, 10) == 0
                    fprintf('Episode %d/%d: 成功率=%.1f%%, 检测率=%.1f%%, RADI=%.4f, 用时=%.2fs\n', ...
                            episode, n_episodes, all_success_rates(end)*100, ...
                            all_detection_rates(end)*100, all_radi_scores(end), episode_time);
                end
            end
            
            total_time = toc(total_start);
            
            % === 性能分析报告 ===
            fprintf('\n=== 性能测试报告 ===\n');
            
            % 时间性能
            fprintf('时间性能:\n');
            fprintf('  总耗时: %.2f秒\n', total_time);
            fprintf('  平均每episode: %.3f秒\n', mean(episode_times));
            fprintf('  平均每step: %.4f秒\n', total_time / (n_episodes * n_steps_per_episode));
            fprintf('  处理速度: %.0f steps/秒\n', (n_episodes * n_steps_per_episode) / total_time);
            
            % 功能性能
            fprintf('\n功能性能:\n');
            final_success_rate = mean(all_success_rates(end-9:end));
            final_detection_rate = mean(all_detection_rates(end-9:end));
            final_radi = mean(all_radi_scores(end-9:end));
            
            fprintf('  最终攻击成功率: %.1f%% (目标: 30-50%%)\n', final_success_rate*100);
            fprintf('  最终检测率: %.1f%% (目标: >40%%)\n', final_detection_rate*100);
            fprintf('  最终RADI: %.4f (越小越好)\n', final_radi);
            
            % 学习趋势
            initial_radi = mean(all_radi_scores(1:5));
            radi_improvement = (initial_radi - final_radi) / initial_radi * 100;
            fprintf('  RADI改善: %.1f%%\n', radi_improvement);
            
            % 稳定性分析
            success_stability = std(all_success_rates(end-9:end));
            detection_stability = std(all_detection_rates(end-9:end));
            radi_stability = std(all_radi_scores(end-9:end));
            
            fprintf('\n稳定性分析:\n');
            fprintf('  成功率稳定性: %.3f (标准差)\n', success_stability);
            fprintf('  检测率稳定性: %.3f (标准差)\n', detection_stability);
            fprintf('  RADI稳定性: %.4f (标准差)\n', radi_stability);
            
            % === 性能评估 ===
            fprintf('\n=== 性能评估 ===\n');
            
            performance_score = 0;
            max_score = 5;
            
            % 评估1: 攻击成功率控制
            if final_success_rate >= 0.3 && final_success_rate <= 0.6
                fprintf('✓ 攻击成功率控制良好\n');
                performance_score = performance_score + 1;
            else
                fprintf('✗ 攻击成功率需要调整 (当前: %.1f%%, 理想: 30-60%%)\n', final_success_rate*100);
            end
            
            % 评估2: 检测率表现
            if final_detection_rate >= 0.3
                fprintf('✓ 检测率表现良好\n');
                performance_score = performance_score + 1;
            else
                fprintf('✗ 检测率偏低 (当前: %.1f%%, 理想: >30%%)\n', final_detection_rate*100);
            end
            
            % 评估3: RADI改善
            if radi_improvement > 5
                fprintf('✓ RADI性能有明显改善\n');
                performance_score = performance_score + 1;
            elseif radi_improvement > 0
                fprintf('◐ RADI性能有轻微改善\n');
                performance_score = performance_score + 0.5;
            else
                fprintf('✗ RADI性能无改善\n');
            end
            
            % 评估4: 系统稳定性
            if success_stability < 0.1 && detection_stability < 0.1
                fprintf('✓ 系统稳定性良好\n');
                performance_score = performance_score + 1;
            else
                fprintf('◐ 系统稳定性一般\n');
                performance_score = performance_score + 0.5;
            end
            
            % 评估5: 计算性能
            steps_per_sec = (n_episodes * n_steps_per_episode) / total_time;
            if steps_per_sec > 1000
                fprintf('✓ 计算性能优秀\n');
                performance_score = performance_score + 1;
            elseif steps_per_sec > 500
                fprintf('◐ 计算性能良好\n');
                performance_score = performance_score + 0.5;
            else
                fprintf('✗ 计算性能需要优化\n');
            end
            
            % 总体评估
            fprintf('\n总体性能评分: %.1f/%d\n', performance_score, max_score);
            
            if performance_score >= 4.5
                fprintf('🎉 性能优秀，环境可以投入使用！\n');
            elseif performance_score >= 3.5
                fprintf('👍 性能良好，可以使用但建议优化\n');
            elseif performance_score >= 2.5
                fprintf('⚠️  性能一般，需要进一步调优\n');
            else
                fprintf('❌ 性能不佳，需要重大改进\n');
            end
            
            fprintf('\n=== 性能测试完成 ===\n');
        end
    end
end