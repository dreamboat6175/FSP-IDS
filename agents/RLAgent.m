%% RLAgent.m - 强化学习智能体基类 (改进版)
% =========================================================================
% 描述: 所有RL智能体的抽象基类，支持灵活的探索策略配置
% =========================================================================

classdef (Abstract) RLAgent < handle
    
    properties (Access = public)
        name                % 智能体名称
        agent_type          % 智能体类型
        state_dim           % 状态空间维度
        action_dim          % 动作空间维度
        config              % 配置参数
        
        % 学习参数
        learning_rate       % 学习率
        discount_factor     % 折扣因子
        
        % 探索策略
        exploration_strategy % 探索策略类型: 'epsilon-greedy', 'softmax', 'none'
        
        % Epsilon-Greedy 参数
        epsilon             % 探索率
        epsilon_min         % 最小探索率
        epsilon_decay       % 探索率衰减
        
        % Softmax/Boltzmann 参数
        temperature         % 温度参数
        temperature_decay   % 温度衰减
        temperature_min     % 最小温度
        
        % 学习率调度
        learning_rate_min   % 最小学习率
        learning_rate_decay % 学习率衰减
        
        % 性能跟踪
        strategy_pool       % 策略池
        pool_size_limit     % 策略池大小限制
        update_count        % 更新计数
        total_reward        % 总奖励
        episode_rewards     % episode奖励历史
        action_history      % 动作历史
        rewards_history      % 奖励历史

         % 新增：RADI相关属性
        radi_history         % RADI历史记录
        previous_radi        % 上一次的RADI值
    end
    
    methods (Abstract)
        % 抽象方法 - 子类必须实现
        action = selectAction(obj, state)
        update(obj, state, action, reward, next_state, next_action)
    end
    
    methods
        function obj = RLAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            
            obj.name = name;
            obj.agent_type = agent_type;
            obj.config = config;
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            
            % 应用配置管理器的合并功能
            if exist('ConfigManager', 'class')
                ConfigManager.mergeAgentConfig(obj, config);
            else
                % 如果ConfigManager不可用，使用传统方式
                obj.applyLegacyConfig(config);
            end
            
            % 初始化性能跟踪
            obj.strategy_pool = {};
            obj.pool_size_limit = 50;
            obj.update_count = 0;
            obj.total_reward = 0;
            obj.episode_rewards = [];
            obj.action_history = [];

            % 初始化RADI相关属性
            obj.radi_history = [];
            obj.previous_radi = 0.1;
            
            % 初始化奖励历史
            obj.rewards_history = [];
        end
        
        function applyLegacyConfig(obj, config)
            % 传统配置方式（向后兼容）
            
            % 基本学习参数
            obj.learning_rate = obj.getConfigValue(config, 'learning_rate', 0.1);
            obj.discount_factor = obj.getConfigValue(config, 'discount_factor', 0.95);
            
            % 探索策略（默认使用epsilon-greedy）
            obj.exploration_strategy = obj.getConfigValue(config, 'exploration_strategy', 'epsilon-greedy');
            
            % Epsilon-Greedy参数
            obj.epsilon = obj.getConfigValue(config, 'epsilon', 0.3);
            obj.epsilon_min = obj.getConfigValue(config, 'epsilon_min', 0.01);
            obj.epsilon_decay = obj.getConfigValue(config, 'epsilon_decay', 0.995);
            
            % Softmax参数
            obj.temperature = obj.getConfigValue(config, 'temperature', 1.0);
            obj.temperature_decay = obj.getConfigValue(config, 'temperature_decay', 0.995);
            obj.temperature_min = obj.getConfigValue(config, 'temperature_min', 0.1);
            
            % 学习率调度
            obj.learning_rate_min = obj.getConfigValue(config, 'learning_rate_min', 0.001);
            obj.learning_rate_decay = obj.getConfigValue(config, 'learning_rate_decay', 0.9995);
        end
        
        function value = getConfigValue(obj, config, field, default)
            % 从配置中获取值，如果不存在则使用默认值
            if isfield(config, field)
                value = config.(field);
            else
                value = default;
            end
        end
        
        function state_idx = encodeState(obj, state_vec)
            % 改进的状态编码：保留更多信息
            
            if isempty(state_vec)
                state_idx = 1;
                return;
            end
            
            % 确保state_vec是向量
            state_vec = reshape(state_vec, 1, []);
            
            % 提取关键特征
            n_stations = 10; % 或从配置获取
            
            if length(state_vec) >= n_stations
                % 提取攻击者策略部分
                attacker_strategy = state_vec(1:n_stations);
                [~, top_threat] = max(attacker_strategy);
                
                % 提取最近攻击频率
                if length(state_vec) >= 2*n_stations
                    recent_freq = state_vec(n_stations+1:2*n_stations);
                    [~, recent_target] = max(recent_freq);
                else
                    recent_target = 1;
                end
                
                % 创建复合索引
                state_idx = (top_threat - 1) * n_stations + recent_target;
            else
                % 简单哈希
                state_hash = sum(state_vec .* (1:length(state_vec)));
                state_idx = mod(round(state_hash * 1000), obj.state_dim) + 1;
            end
            
            % 确保索引在有效范围内
            state_idx = max(1, min(obj.state_dim, state_idx));
        end      
        
        function action = exploreAction(obj, greedy_action, action_space)
            % 根据探索策略选择动作
            
            switch obj.exploration_strategy
                case 'epsilon-greedy'
                    action = obj.epsilonGreedyExplore(greedy_action, action_space);
                    
                case 'softmax'
                    action = obj.softmaxExplore(action_space);
                    
                case 'none'
                    action = greedy_action;
                    
                otherwise
                    % 默认使用epsilon-greedy
                    action = obj.epsilonGreedyExplore(greedy_action, action_space);
            end
        end
        
        function action = epsilonGreedyExplore(obj, greedy_action, action_space)
            % Epsilon-Greedy探索
            if rand() < obj.epsilon
                % 探索：随机选择动作
                if isscalar(action_space)
                    action = randi(action_space);
                else
                    action = action_space(randi(length(action_space)));
                end
            else
                % 利用：选择贪婪动作
                action = greedy_action;
            end
        end
        
        function action = softmaxExplore(obj, q_values)
            % Softmax/Boltzmann探索
            % q_values: 当前状态的Q值向量
            
            if obj.temperature <= 0
                % 温度为0时，选择最大Q值
                [~, action] = max(q_values);
            else
                % 使用Boltzmann分布
                scaled_q = q_values / obj.temperature;
                % 数值稳定的softmax
                exp_q = exp(scaled_q - max(scaled_q));
                probs = exp_q / sum(exp_q);
                
                % 根据概率分布选择动作
                action = randsample(1:length(q_values), 1, true, probs);
            end
        end
        
        function recordAction(obj, state, action)
            % 记录动作历史
            try
                state_idx = obj.encodeState(state);
                obj.action_history(end+1, :) = [state_idx, action];
            catch
                % 如果记录失败，跳过
            end
        end
        
        function reset(obj)
            % 重置智能体（可被子类覆盖）
            % 默认实现：不做任何操作
        end
        
        function resetEpisode(obj)
            % 重置episode相关的状态
            obj.updateExplorationParameters();
        end
        
        function info = getInfo(obj)
            % 获取智能体信息
            info = struct();
            info.name = obj.name;
            info.type = obj.agent_type;
            info.learning_rate = obj.learning_rate;
            info.exploration_strategy = obj.exploration_strategy;
            
            % 根据探索策略添加相关参数
            switch obj.exploration_strategy
                case 'epsilon-greedy'
                    info.epsilon = obj.epsilon;
                    info.epsilon_min = obj.epsilon_min;
                    info.epsilon_decay = obj.epsilon_decay;
                    
                case 'softmax'
                    info.temperature = obj.temperature;
                    info.temperature_min = obj.temperature_min;
                    info.temperature_decay = obj.temperature_decay;
            end
            
            info.state_dim = obj.state_dim;
            info.action_dim = obj.action_dim;
        end
        
        function updateParameters(obj)
            % 更新学习参数（衰减等）
            
            % 更新学习率
            obj.learning_rate = max(obj.learning_rate_min, ...
                                   obj.learning_rate * obj.learning_rate_decay);
            
            % 更新探索参数
            obj.updateExplorationParameters();
        end
        
        function updateExplorationParameters(obj)
            % 更新探索参数
            
            switch obj.exploration_strategy
                case 'epsilon-greedy'
                    % 更新epsilon
                    obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
                    
                case 'softmax'
                    % 更新温度
                    obj.temperature = max(obj.temperature_min, ...
                                         obj.temperature * obj.temperature_decay);
            end
        end
        
        function decay(obj)
            % 参数衰减方法（与updateParameters相同，提供别名）
            obj.updateParameters();
        end
        
        function policy = getPolicy(obj)
            % 获取当前策略 - 默认实现
            % 子类应该覆盖此方法以提供具体的策略
            policy = ones(1, obj.action_dim) / obj.action_dim;
        end
        
        function strategy = getStrategy(obj)
            % 获取当前策略分布（与getPolicy相同）
            strategy = obj.getPolicy();
        end
        
        function stats = getStats(obj)
            % 获取统计信息
            stats = struct();
            stats.total_reward = obj.total_reward;
            stats.update_count = obj.update_count;
            stats.episode_rewards = obj.episode_rewards;
            
            % 探索统计
            switch obj.exploration_strategy
                case 'epsilon-greedy'
                    stats.current_epsilon = obj.epsilon;
                case 'softmax'
                    stats.current_temperature = obj.temperature;
            end
            
            % 动作分布统计
            if ~isempty(obj.action_history)
                actions = obj.action_history(:, 2);
                stats.action_distribution = histcounts(actions, 1:obj.action_dim+1) / length(actions);
            else
                stats.action_distribution = [];
            end
        end
        function radi = calculateRADI(obj)
            % 统一的RADI计算方法
            
            % 初始化历史记录
            if ~isfield(obj, 'radi_history') || isempty(obj.radi_history)
                obj.radi_history = [];
            end
            
            if ~isfield(obj, 'previous_radi')
                obj.previous_radi = 0.1;
            end
            
            % 基于Q表计算RADI
            if ~isempty(obj.Q_table)
                % 只考虑已访问的状态-动作对
                visited_mask = obj.visit_count > 0;
                if any(visited_mask(:))
                    q_values = obj.Q_table(visited_mask);
                    
                    % 计算Q值的统计特征
                    q_std = std(q_values);
                    q_mean = mean(abs(q_values));
                    
                    % RADI = 标准差 / (1 + 平均值)
                    current_radi = q_std / (1 + q_mean);
                    
                    % 使用指数移动平均平滑
                    smoothing_factor = 0.9;
                    radi = smoothing_factor * obj.previous_radi + (1 - smoothing_factor) * current_radi;
                    
                    obj.previous_radi = radi;
                else
                    % 未访问任何状态时的默认值
                    radi = 0.1;
                end
            else
                % Q表为空时的默认值
                radi = 0.1;
            end
            
            % 确保RADI在合理范围内
            radi = max(0.01, min(1.0, radi));
            
            % 记录历史
            obj.radi_history(end+1) = radi;
            
            % 保持历史记录长度
            if length(obj.radi_history) > 1000
                obj.radi_history = obj.radi_history(end-999:end);
            end
        end
        function updateEpsilon(obj)
            % 只在一定更新次数后才开始衰减
            if obj.update_count > 2000
                obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
            end
            
            % 防止epsilon过低
            if obj.epsilon < obj.epsilon_min
                obj.epsilon = obj.epsilon_min;
            end
            
            % 周期性提升探索（可选）
            if mod(obj.update_count, 5000) == 0 && obj.update_count > 0
                obj.epsilon = min(1.0, obj.epsilon * 1.2);
                fprintf('[INFO] %s - 周期性提升探索率至: %.3f\n', obj.name, obj.epsilon);
            end
        end
        function updateExplorationParams(obj)
            % 更新探索参数（epsilon和temperature衰减）
            
            % 更新epsilon
            if obj.epsilon > obj.epsilon_min
                obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
            end
            
            % 更新temperature  
            if obj.temperature > obj.temperature_min
                obj.temperature = max(obj.temperature_min, obj.temperature * obj.temperature_decay);
            end
        end
        
        function initializeFSPComponents(obj)
            % 初始化FSP双核大脑组件（仅用于防御者）
            
            if ~obj.is_dual_brain
                return;
            end
            
            % 分析员大脑初始化
            obj.analyst_brain.attacker_avg_strategy = ones(1, obj.action_dim) / obj.action_dim;
            obj.analyst_brain.alpha_ewma = 0.1; % EWMA学习率
            obj.analyst_brain.observation_history = [];
            obj.analyst_brain.strategy_estimates = [];
            
            % 战略家大脑初始化
            obj.strategist_brain.optimal_responses = [];
            obj.strategist_brain.response_history = [];
            obj.strategist_brain.confidence_scores = [];
            
            % 自适应alpha参数
            obj.adaptive_alpha = true;
            obj.alpha_min = 0.05;
            obj.alpha_max = 0.3;
        end
        
        function deployment = computeOptimalDefenseDeployment(obj, attacker_strategy, state)
            % 计算针对攻击者平均策略的最优防御部署
            
            % 获取站点价值
            if isfield(obj.config, 'station_values')
                station_values = obj.config.station_values;
            elseif isfield(obj.config, 'system') && isfield(obj.config.system, 'station_values')
                station_values = obj.config.system.station_values;
            else
                station_values = ones(1, obj.action_dim);
            end
            
            % 获取总资源
            if isfield(obj.config, 'total_resources')
                total_resources = obj.config.total_resources;
            elseif isfield(obj.config, 'system') && isfield(obj.config.system, 'total_resources')
                total_resources = obj.config.system.total_resources;
            else
                total_resources = 100;
            end
            
            % 基于攻击者策略和站点价值计算防御部署
            risk_scores = attacker_strategy .* station_values;
            
            % 归一化风险分数并分配资源
            if sum(risk_scores) > 0
                deployment = (risk_scores / sum(risk_scores)) * total_resources;
            else
                deployment = ones(1, obj.action_dim) * (total_resources / obj.action_dim);
            end
            
            % 确保最小分配
            min_allocation = total_resources * 0.01; % 1%的最小分配
            deployment = max(min_allocation, deployment);
            
            % 重新归一化
            deployment = deployment * total_resources / sum(deployment);
        end
    end
end