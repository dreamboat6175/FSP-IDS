%% QLearningAgent.m - Q-Learning智能体实现
% =========================================================================
% 描述: 实现标准Q-Learning算法的智能体
% =========================================================================

classdef QLearningAgent < RLAgent
    
    properties
        Q_table    % Q值表
        visit_count % 状态-动作访问计数
        lr_scheduler % 学习率调度器
    end
    
    methods
        function obj = QLearningAgent(name, agent_type, config, state_dim, action_dim)
    % 构造函数
    obj@RLAgent(name, agent_type, config, state_dim, action_dim);
    
    % 使用稀疏矩阵优化内存
    if state_dim * action_dim > 1e6
        obj.Q_table = sparse(state_dim, action_dim);
        obj.visit_count = sparse(state_dim, action_dim);
    else
        obj.Q_table = zeros(state_dim, action_dim);
        obj.visit_count = zeros(state_dim, action_dim);
    end
    
    % 优化的初始化策略
    % 使用较高的初始值鼓励探索（乐观初始化）
    initial_value = 5.0; % 提高初始值
    noise_level = 0.5;   % 增加噪声
    
    if issparse(obj.Q_table)
        % 稀疏矩阵的初始化
        [rows, cols] = size(obj.Q_table);
        init_indices = randi([1, rows*cols], [1, min(1000, rows*cols/10)]);
        obj.Q_table(init_indices) = initial_value + randn(size(init_indices)) * noise_level;
    else
        % 密集矩阵的初始化
        obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                      randn(state_dim, action_dim) * noise_level;
    end
    
    % 添加学习率调度器
    obj.lr_scheduler = struct();
    obj.lr_scheduler.initial_lr = config.learning_rate;
    obj.lr_scheduler.min_lr = 0.001;
    obj.lr_scheduler.decay_steps = 1000;
end

% 修改 update 方法，添加自适应学习率：

function update(obj, state_vec, action, reward, next_state_vec, ~)
    % 改进的Q-Learning更新
    
    state = obj.getStateIndex(state_vec);
    next_state = obj.getStateIndex(next_state_vec);
    
    % 获取最大Q值
    max_next_q = max(obj.Q_table(next_state, :));
    
    % 计算TD误差
    current_q = obj.Q_table(state, action);
    td_error = reward + obj.discount_factor * max_next_q - current_q;
    
    % 自适应学习率
    visit_count = obj.visit_count(state, action);
    adaptive_lr = obj.lr_scheduler.initial_lr / (1 + 0.01 * visit_count);
    adaptive_lr = max(adaptive_lr, obj.lr_scheduler.min_lr);
    
    % 更新Q值
    obj.Q_table(state, action) = current_q + adaptive_lr * td_error;
    
    % 更新访问计数
    obj.visit_count(state, action) = visit_count + 1;
    
    % 记录性能
    obj.recordReward(reward);
    obj.update_count = obj.update_count + 1;
    
    % 定期调整探索率
    if mod(obj.update_count, 100) == 0
        obj.updateEpsilon();
    end
end

function action = selectAction(obj, state_vec)
            % 选择动作
            
            % 获取状态索引
            state = obj.getStateIndex(state_vec);
            
            % 确保状态索引在有效范围内
            if state < 1 || state > obj.state_dim
                warning('状态索引超出范围: %d (范围: 1-%d)', state, obj.state_dim);
                state = mod(state - 1, obj.state_dim) + 1;
            end
            
            % 获取当前状态的Q值
            q_values = obj.Q_table(state, :);
            
            % 根据策略选择动作
            if obj.use_softmax
                action = obj.boltzmannAction(state, q_values);
            else
                action = obj.epsilonGreedyAction(state, q_values);
            end
            
            % 记录动作
            obj.recordAction(state, action);
        end
        
        function policy = getPolicy(obj)
            % 获取当前策略（Q表）
            policy = obj.Q_table;
        end
        
        function save(obj, filename)
            % 保存模型
            if nargin < 2
                filename = sprintf('models/qlearning_%s_%s.mat', ...
                                 obj.agent_type, datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            % 确保目录存在
            [filepath, ~, ~] = fileparts(filename);
            if ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            
            % 保存数据
            save_data.Q_table = obj.Q_table;
            save_data.visit_count = obj.visit_count;
            save_data.name = obj.name;
            save_data.agent_type = obj.agent_type;
            save_data.update_count = obj.update_count;
            save_data.total_reward = obj.total_reward;
            save_data.episode_rewards = obj.episode_rewards;
            save_data.learning_rate = obj.learning_rate;
            save_data.discount_factor = obj.discount_factor;
            save_data.epsilon = obj.epsilon;
            
            save(filename, 'save_data');
            fprintf('Q-Learning模型已保存: %s\n', filename);
        end
        
        function load(obj, filename)
            % 加载模型
            if exist(filename, 'file')
                load_data = load(filename);
                save_data = load_data.save_data;
                
                obj.Q_table = save_data.Q_table;
                obj.visit_count = save_data.visit_count;
                obj.name = save_data.name;
                obj.agent_type = save_data.agent_type;
                obj.update_count = save_data.update_count;
                obj.total_reward = save_data.total_reward;
                obj.episode_rewards = save_data.episode_rewards;
                obj.learning_rate = save_data.learning_rate;
                obj.discount_factor = save_data.discount_factor;
                obj.epsilon = save_data.epsilon;
                
                fprintf('Q-Learning模型已加载: %s\n', filename);
            else
                error('模型文件不存在: %s', filename);
            end
        end
        
        function visualizeQTable(obj, num_states, num_actions)
            % 可视化Q表（显示部分）
            if nargin < 2
                num_states = min(50, obj.state_dim);
            end
            if nargin < 3
                num_actions = min(20, obj.action_dim);
            end
            
            figure('Name', sprintf('%s Q-Table Visualization', obj.name));
            
            % Q值热力图
            subplot(2, 1, 1);
            imagesc(obj.Q_table(1:num_states, 1:num_actions));
            colorbar;
            xlabel('动作');
            ylabel('状态');
            title(sprintf('%s Q值热力图', obj.name));
            colormap('hot');
            
            % 访问频率热力图
            subplot(2, 1, 2);
            imagesc(log1p(obj.visit_count(1:num_states, 1:num_actions)));
            colorbar;
            xlabel('动作');
            ylabel('状态');
            title('状态-动作访问频率（对数尺度）');
            colormap('cool');
        end
        
        function metrics = getDetailedMetrics(obj)
            % 获取详细的性能指标
            metrics = obj.getStatistics();
            
            % Q值统计
            metrics.avg_q_value = mean(obj.Q_table(:));
            metrics.max_q_value = max(obj.Q_table(:));
            metrics.min_q_value = min(obj.Q_table(:));
            metrics.q_value_std = std(obj.Q_table(:));
            
            % 探索统计
            total_visits = sum(obj.visit_count(:));
            if total_visits > 0
                visit_probs = obj.visit_count(:) / total_visits;
                metrics.exploration_entropy = -sum(visit_probs(visit_probs>0) .* ...
                                                 log(visit_probs(visit_probs>0)));
            else
                metrics.exploration_entropy = 0;
            end
            
            % 策略稳定性
            if ~isempty(obj.action_history) && size(obj.action_history, 1) > 10
                recent_actions = obj.action_history(end-9:end, 2);
                metrics.action_diversity = length(unique(recent_actions)) / 10;
            else
                metrics.action_diversity = 0;
            end
        end
    end
end