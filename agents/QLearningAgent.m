%% QLearningAgent.m - 优化的Q-Learning智能体实现
% =========================================================================
% 描述: 实现标准Q-Learning算法的智能体，优化了可维护性和性能
% 优化要点：
% 1. 参数验证和错误处理
% 2. 内存优化（稀疏矩阵支持）
% 3. 模块化设计
% 4. 性能监控
% 5. 数据安全性
% =========================================================================

classdef QLearningAgent < RLAgent
    
    properties (Access = private)
        Q_table          % Q值表
        visit_count      % 状态-动作访问计数
        lr_scheduler     % 学习率调度器
        exploration_strategy % 探索策略
    end
    
    properties (Access = public)
        statistics       % 性能统计信息
    end
    
    methods (Access = public)
        function obj = QLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数 - 安全的参数处理
            
            % 调用父类构造函数
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 验证输入参数
            obj.validateInputs(state_dim, action_dim, config);
            
            % 初始化Q表
            obj.initializeQTable(state_dim, action_dim);
            
            % 初始化学习率调度器
            obj.initializeLearningRateScheduler(config);
            
            % 初始化探索策略
            obj.initializeExplorationStrategy(config);
            
            % 初始化统计信息
            obj.initializeStatistics();
            
            % 设置默认参数
            obj.setDefaultParameters();
        end
        
        function action = selectAction(obj, state)
            % 选择动作 - 使用epsilon-greedy策略
            
            try
                % 输入验证
                state = obj.validateAndPreprocessState(state);
                
                % 获取状态索引
                state_idx = obj.encodeState(state);
                
                % 选择动作
                if rand() < obj.epsilon
                    % 探索：随机选择
                    action = obj.selectRandomAction();
                else
                    % 利用：选择最优动作
                    action = obj.selectGreedyAction(state_idx);
                end
                
                % 记录动作选择
                obj.recordActionSelection(state_idx, action);
                
                % 更新统计信息
                obj.updateActionStatistics(action);
                
            catch ME
                warning('动作选择失败，使用随机动作: %s', ME.message);
                action = obj.selectRandomAction();
            end
        end
        
        function update(obj, state, action, reward, next_state, ~)
            % Q值更新 - 标准Q-learning算法
            
            try
                % 输入验证和预处理
                [state, action, reward, next_state] = obj.validateUpdateInputs(...
                    state, action, reward, next_state);
                
                % 编码状态
                state_idx = obj.encodeState(state);
                next_state_idx = obj.encodeState(next_state);
                action_idx = obj.encodeAction(action);
                
                % Q-learning更新
                current_q = obj.getQValue(state_idx, action_idx);
                max_next_q = obj.getMaxQValue(next_state_idx);
                
                % 计算目标Q值
                target_q = reward + obj.discount_factor * max_next_q;
                
                % 计算TD误差
                td_error = target_q - current_q;
                
                % 获取当前学习率
                current_lr = obj.getCurrentLearningRate(state_idx, action_idx);
                
                % 更新Q值
                new_q = current_q + current_lr * td_error;
                obj.setQValue(state_idx, action_idx, new_q);
                
                % 更新访问计数
                obj.updateVisitCount(state_idx, action_idx);
                
                % 更新学习率调度器
                obj.updateLearningRateScheduler();
                
                % 更新探索参数
                obj.updateExplorationParameters();
                
                % 记录学习统计
                obj.recordLearningStatistics(reward, td_error);
                
            catch ME
                warning('Q值更新失败: %s', ME.message);
            end
        end
        
        function stats = getStatistics(obj)
            % 获取智能体统计信息
            
            stats = obj.statistics;
            
            % 添加Q值统计
            if ~isempty(obj.Q_table)
                stats.q_values = struct();
                stats.q_values.mean = mean(obj.Q_table(:));
                stats.q_values.std = std(obj.Q_table(:));
                stats.q_values.max = max(obj.Q_table(:));
                stats.q_values.min = min(obj.Q_table(:));
            end
            
            % 添加探索统计
            if obj.statistics.total_actions > 0
                stats.exploration_rate = obj.statistics.exploration_actions / obj.statistics.total_actions;
            else
                stats.exploration_rate = 0;
            end
        end
        
        function saveModel(obj, filename)
            % 保存模型到文件
            
            try
                model_data = struct();
                model_data.Q_table = obj.Q_table;
                model_data.visit_count = obj.visit_count;
                model_data.statistics = obj.statistics;
                model_data.parameters = obj.getParameters();
                model_data.timestamp = datetime('now');
                
                save(filename, 'model_data');
                
            catch ME
                error('模型保存失败: %s', ME.message);
            end
        end
        
        function loadModel(obj, filename)
            % 从文件加载模型
            
            try
                if exist(filename, 'file')
                    loaded = load(filename);
                    model_data = loaded.model_data;
                    
                    obj.Q_table = model_data.Q_table;
                    obj.visit_count = model_data.visit_count;
                    obj.statistics = model_data.statistics;
                    obj.setParameters(model_data.parameters);
                    
                else
                    error('模型文件不存在: %s', filename);
                end
                
            catch ME
                error('模型加载失败: %s', ME.message);
            end
        end
        
        function visualizeQTable(obj, varargin)
            % 可视化Q表
            
            p = inputParser;
            addParameter(p, 'max_states', 50, @isnumeric);
            addParameter(p, 'max_actions', 20, @isnumeric);
            parse(p, varargin{:});
            
            max_states = min(p.Results.max_states, size(obj.Q_table, 1));
            max_actions = min(p.Results.max_actions, size(obj.Q_table, 2));
            
            figure('Name', sprintf('%s Q-Table可视化', obj.name));
            
            % Q值热力图
            subplot(2, 1, 1);
            imagesc(obj.Q_table(1:max_states, 1:max_actions));
            colorbar;
            xlabel('动作');
            ylabel('状态');
            title('Q值热力图');
            colormap('hot');
            
            % 访问频率热力图
            subplot(2, 1, 2);
            if ~isempty(obj.visit_count)
                imagesc(log1p(obj.visit_count(1:max_states, 1:max_actions)));
                colorbar;
                xlabel('动作');
                ylabel('状态');
                title('访问频率热力图（对数尺度）');
                colormap('cool');
            end
        end
    end
    
    methods (Access = protected)
        function initializeStatistics(obj)
            % 初始化统计信息
            
            obj.statistics = struct();
            obj.statistics.total_updates = 0;
            obj.statistics.total_actions = 0;
            obj.statistics.exploration_actions = 0;
            obj.statistics.total_reward = 0;
            obj.statistics.episode_rewards = [];
            obj.statistics.td_errors = [];
            obj.statistics.learning_rates = [];
        end
    end
    
    methods (Access = private)
        function validateInputs(obj, state_dim, action_dim, config)
            % 验证构造函数输入
            
            assert(state_dim > 0, 'state_dim必须大于0');
            assert(action_dim > 0, 'action_dim必须大于0');
            assert(isstruct(config), 'config必须是结构体');
            
            % 验证必需的配置字段
            required_fields = {'learning_rate', 'discount_factor', 'epsilon'};
            for i = 1:length(required_fields)
                assert(isfield(config, required_fields{i}), ...
                    '配置缺少必需字段: %s', required_fields{i});
            end
        end
        
        function initializeQTable(obj, state_dim, action_dim)
            % 初始化Q表 - 支持稀疏矩阵优化
            
            % 使用稀疏矩阵优化大状态空间
            if state_dim * action_dim > 1e6
                obj.Q_table = sparse(state_dim, action_dim);
                obj.visit_count = sparse(state_dim, action_dim);
            else
                obj.Q_table = zeros(state_dim, action_dim);
                obj.visit_count = zeros(state_dim, action_dim);
            end
            
            % 乐观初始化策略
            initial_value = 1.0;
            noise_level = 0.1;
            
            if issparse(obj.Q_table)
                % 稀疏矩阵的随机初始化
                num_init = min(1000, round(state_dim * action_dim * 0.01));
                init_indices = randi([1, state_dim * action_dim], [1, num_init]);
                obj.Q_table(init_indices) = initial_value + randn(size(init_indices)) * noise_level;
            else
                % 密集矩阵的初始化
                obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                              randn(state_dim, action_dim) * noise_level;
            end
        end
        
        function initializeLearningRateScheduler(obj, config)
            % 初始化学习率调度器
            
            obj.lr_scheduler = struct();
            obj.lr_scheduler.initial_lr = config.learning_rate;
            obj.lr_scheduler.current_lr = config.learning_rate;
            obj.lr_scheduler.min_lr = 0.001;
            obj.lr_scheduler.decay_rate = 0.9995;
            obj.lr_scheduler.decay_steps = 1000;
            obj.lr_scheduler.step_count = 0;
            
            % 从配置读取可选参数
            if isfield(config, 'learning_rate_min')
                obj.lr_scheduler.min_lr = config.learning_rate_min;
            end
            if isfield(config, 'learning_rate_decay')
                obj.lr_scheduler.decay_rate = config.learning_rate_decay;
            end
        end
        
        function initializeExplorationStrategy(obj, config)
            % 初始化探索策略
            
            obj.exploration_strategy = struct();
            obj.exploration_strategy.type = 'epsilon_greedy';
            obj.exploration_strategy.epsilon_min = 0.01;
            obj.exploration_strategy.epsilon_decay = 0.995;
            
            % 从配置读取参数
            if isfield(config, 'epsilon_min')
                obj.exploration_strategy.epsilon_min = config.epsilon_min;
            end
            if isfield(config, 'epsilon_decay')
                obj.exploration_strategy.epsilon_decay = config.epsilon_decay;
            end
        end
        
        function setDefaultParameters(obj)
            % 设置默认参数
            
            if isempty(obj.epsilon)
                obj.epsilon = 0.1;
            end
            if isempty(obj.learning_rate)
                obj.learning_rate = 0.1;
            end
            if isempty(obj.discount_factor)
                obj.discount_factor = 0.95;
            end
        end
        
        function state = validateAndPreprocessState(obj, state)
            % 验证和预处理状态
            
            if isempty(state)
                warning('状态为空，使用零向量');
                state = zeros(1, obj.state_dim);
            end
            
            % 确保正确的形状
            state = reshape(state, 1, []);
            
            % 检查维度
            if length(state) ~= obj.state_dim
                if length(state) < obj.state_dim
                    % 补零
                    state = [state, zeros(1, obj.state_dim - length(state))];
                else
                    % 截断
                    state = state(1:obj.state_dim);
                end
            end
        end
        
        function state_idx = encodeState(obj, state)
            % 状态编码 - 简单的哈希方法
            
            % 简单的线性编码
            state_idx = 1;
            for i = 1:length(state)
                state_idx = state_idx + round(state(i) * 100) * (i^2);
            end
            state_idx = mod(abs(state_idx), size(obj.Q_table, 1)) + 1;
        end
        
        function action_idx = encodeAction(obj, action)
            % 动作编码
            
            if isscalar(action)
                action_idx = max(1, min(action, size(obj.Q_table, 2)));
            else
                % 多维动作的简单编码
                action_idx = mod(sum(action), size(obj.Q_table, 2)) + 1;
            end
        end
        
        function action = selectRandomAction(obj)
            % 随机选择动作
            
            action = randi(obj.action_dim);
            obj.statistics.exploration_actions = obj.statistics.exploration_actions + 1;
        end
        
        function action = selectGreedyAction(obj, state_idx)
            % 贪心选择动作
            
            q_values = obj.Q_table(state_idx, :);
            [~, action] = max(q_values);
        end
        
        function q_value = getQValue(obj, state_idx, action_idx)
            % 获取Q值
            
            q_value = obj.Q_table(state_idx, action_idx);
        end
        
        function setQValue(obj, state_idx, action_idx, value)
            % 设置Q值
            
            obj.Q_table(state_idx, action_idx) = value;
        end
        
        function max_q = getMaxQValue(obj, state_idx)
            % 获取状态的最大Q值
            
            max_q = max(obj.Q_table(state_idx, :));
        end
        
        function lr = getCurrentLearningRate(obj, state_idx, action_idx)
            % 获取当前学习率（可以基于访问次数自适应）
            
            if obj.visit_count(state_idx, action_idx) > 0
                % 基于访问次数的自适应学习率
                visit_count = obj.visit_count(state_idx, action_idx);
                lr = obj.lr_scheduler.current_lr / (1 + visit_count * 0.01);
            else
                lr = obj.lr_scheduler.current_lr;
            end
            
            lr = max(lr, obj.lr_scheduler.min_lr);
        end
        
        function updateVisitCount(obj, state_idx, action_idx)
            % 更新访问计数
            
            obj.visit_count(state_idx, action_idx) = obj.visit_count(state_idx, action_idx) + 1;
        end
        
        function updateLearningRateScheduler(obj)
            % 更新学习率调度器
            
            obj.lr_scheduler.step_count = obj.lr_scheduler.step_count + 1;
            
            if mod(obj.lr_scheduler.step_count, obj.lr_scheduler.decay_steps) == 0
                obj.lr_scheduler.current_lr = max(...
                    obj.lr_scheduler.current_lr * obj.lr_scheduler.decay_rate, ...
                    obj.lr_scheduler.min_lr);
            end
        end
        
        function updateExplorationParameters(obj)
            % 更新探索参数
            
            obj.epsilon = max(...
                obj.epsilon * obj.exploration_strategy.epsilon_decay, ...
                obj.exploration_strategy.epsilon_min);
        end
        
        function recordActionSelection(obj, state_idx, action)
            % 记录动作选择
            
            obj.statistics.total_actions = obj.statistics.total_actions + 1;
        end
        
        function updateActionStatistics(obj, action)
            % 更新动作统计
            
            % 可以在这里添加特定的动作统计逻辑
        end
        
        function [state, action, reward, next_state] = validateUpdateInputs(obj, state, action, reward, next_state)
            % 验证更新输入
            
            state = obj.validateAndPreprocessState(state);
            next_state = obj.validateAndPreprocessState(next_state);
            
            if isempty(action)
                warning('动作为空，使用默认值1');
                action = 1;
            end
            
            if ~isnumeric(reward) || ~isscalar(reward)
                warning('奖励格式不正确，使用默认值0');
                reward = 0;
            end
        end
        
        function recordLearningStatistics(obj, reward, td_error)
            % 记录学习统计
            
            obj.statistics.total_updates = obj.statistics.total_updates + 1;
            obj.statistics.total_reward = obj.statistics.total_reward + reward;
            obj.statistics.td_errors = [obj.statistics.td_errors; td_error];
            obj.statistics.learning_rates = [obj.statistics.learning_rates; obj.lr_scheduler.current_lr];
            
            % 限制历史记录长度，避免内存溢出
            max_history = 10000;
            if length(obj.statistics.td_errors) > max_history
                obj.statistics.td_errors = obj.statistics.td_errors(end-max_history+1:end);
            end
            if length(obj.statistics.learning_rates) > max_history
                obj.statistics.learning_rates = obj.statistics.learning_rates(end-max_history+1:end);
            end
        end
        
        function params = getParameters(obj)
            % 获取当前参数
            
            params = struct();
            params.learning_rate = obj.learning_rate;
            params.discount_factor = obj.discount_factor;
            params.epsilon = obj.epsilon;
            params.lr_scheduler = obj.lr_scheduler;
            params.exploration_strategy = obj.exploration_strategy;
        end
        
        function setParameters(obj, params)
            % 设置参数
            
            if isfield(params, 'learning_rate')
                obj.learning_rate = params.learning_rate;
            end
            if isfield(params, 'discount_factor')
                obj.discount_factor = params.discount_factor;
            end
            if isfield(params, 'epsilon')
                obj.epsilon = params.epsilon;
            end
            if isfield(params, 'lr_scheduler')
                obj.lr_scheduler = params.lr_scheduler;
            end
            if isfield(params, 'exploration_strategy')
                obj.exploration_strategy = params.exploration_strategy;
            end
        end
    end
end