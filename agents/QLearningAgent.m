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
    
    % === 直接赋默认值（这些属性已在基类声明） ===
    obj.epsilon_min = 0.01;
    obj.epsilon_decay = 0.995;
    obj.temperature = 1.0;
    obj.temperature_decay = 0.995;
    obj.learning_rate_min = 0.01;
    obj.learning_rate_decay = 0.9995;
end

% 修改 update 方法，添加自适应学习率：

function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
    % --- Robust shape check ---
    if isempty(action_vec)
        warning('QLearningAgent.update: action_vec is empty, auto-fixing...');
        action_vec = ones(1, 5);
    end
    action_vec = reshape(action_vec, 1, []);
    if isempty(state_vec)
        warning('QLearningAgent.update: state_vec is empty, auto-fixing...');
        state_vec = ones(1, obj.state_dim);
    end
    state_vec = reshape(state_vec, 1, []);
    if ~isempty(next_state_vec)
        next_state_vec = reshape(next_state_vec, 1, []);
    end
    
    % 获取状态索引
    state_idx = obj.getStateIndex(mean(state_vec));
    next_state_idx = obj.getStateIndex(mean(next_state_vec));
    
    % 将站点级动作转换为Q表索引
    n_stations = length(action_vec);
    n_resource_types = obj.action_dim / n_stations;
    
    % 使用第一个站点的动作作为主要索引
    primary_station = 1;
    resource_type = action_vec(primary_station);
    resource_type = max(1, min(n_resource_types, round(resource_type)));
    q_action_idx = (primary_station - 1) * n_resource_types + resource_type;
    q_action_idx = max(1, min(obj.action_dim, q_action_idx));
    
    % 计算TD误差
    current_q = obj.Q_table(state_idx, q_action_idx);
    if isempty(next_action_vec)
        max_next_q = max(obj.Q_table(next_state_idx, :));
    else
        % 处理下一个动作
        next_resource_type = next_action_vec(primary_station);
        next_resource_type = max(1, min(n_resource_types, round(next_resource_type)));
        next_q_action_idx = (primary_station - 1) * n_resource_types + next_resource_type;
        next_q_action_idx = max(1, min(obj.action_dim, next_q_action_idx));
        max_next_q = obj.Q_table(next_state_idx, next_q_action_idx);
    end
    
    td_error = reward + obj.discount_factor * max_next_q - current_q;
    obj.Q_table(state_idx, q_action_idx) = current_q + obj.learning_rate * td_error;
    obj.visit_count(state_idx, q_action_idx) = obj.visit_count(state_idx, q_action_idx) + 1;
    
    obj.recordReward(reward);
    obj.update_count = obj.update_count + 1;
end

        function action_vec = selectAction(obj, state_vec)
    % 改进的动作选择，增加多样性
    
    % 健壮性检查
    if isempty(state_vec)
        state_vec = ones(1, obj.state_dim);
    end
    state_vec = reshape(state_vec, 1, []);
    
    % 获取状态索引
    state_idx = obj.getStateIndex(mean(state_vec));
    
    % 获取Q值
    q_values = obj.Q_table(state_idx, :);
    
    % 确保Q值有效
    if any(isnan(q_values)) || any(isinf(q_values))
        q_values = randn(size(q_values)) * 0.1;
    end
    
    % 动态调整参数
    if isprop(obj, 'epsilon_decay')
        obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
    end
    if isprop(obj, 'temperature_decay') && obj.use_softmax
        obj.temperature = max(0.1, obj.temperature * obj.temperature_decay);
    end
    if isprop(obj, 'learning_rate_decay')
        obj.learning_rate = max(obj.learning_rate_min, obj.learning_rate * obj.learning_rate_decay);
    end
    
    % 获取访问统计
    visit_counts = obj.visit_count(state_idx, :);
    total_visits = sum(visit_counts);
    
    % 自适应探索率
    if total_visits < 10
        effective_epsilon = min(0.9, obj.epsilon * 2);
    elseif total_visits < 50
        effective_epsilon = min(0.7, obj.epsilon * 1.5);
    else
        effective_epsilon = obj.epsilon;
    end
    
    % 探索 vs 利用
    if rand() < effective_epsilon
        % 探索：使用多种策略
        strategy = randi(4);
        
        switch strategy
            case 1
                % 完全随机
                action_vec = rand(1, obj.action_dim);
                
            case 2
                % 基于访问次数的探索
                exploration_weights = 1 ./ (visit_counts + 1);
                action_vec = exploration_weights / sum(exploration_weights);
                % 添加噪声
                action_vec = action_vec + randn(1, obj.action_dim) * 0.1;
                
            case 3
                % 集中式探索
                n_focus = randi([1, min(3, obj.action_dim)]);
                focus_actions = randperm(obj.action_dim, n_focus);
                action_vec = ones(1, obj.action_dim) * 0.05;
                action_vec(focus_actions) = (0.85 + rand() * 0.1) / n_focus;
                
            case 4
                % Thompson采样风格的探索
                sampled_values = q_values + randn(size(q_values)) * std(q_values);
                action_vec = exp(sampled_values) / sum(exp(sampled_values));
        end
        
    else
        % 利用：基于Q值
        if obj.use_softmax
            % Softmax策略
            q_normalized = q_values - max(q_values);  % 避免数值溢出
            exp_values = exp(q_normalized / obj.temperature);
            action_vec = exp_values / sum(exp_values);
            
            % 添加小量噪声
            noise = randn(1, obj.action_dim) * 0.05;
            action_vec = action_vec + noise;
            
        else
            % 改进的ε-greedy
            % 使用UCB来打破平局
            c = 2;  % UCB常数
            ucb_bonus = sqrt(c * log(total_visits + 1) ./ (visit_counts + 1));
            ucb_values = q_values + ucb_bonus;
            
            [~, best_action] = max(ucb_values);
            
            % 不使用纯one-hot，而是主导动作+其他小概率
            action_vec = ones(1, obj.action_dim) * 0.05;
            action_vec(best_action) = 0.75;
            
            % 给相似Q值的动作更多概率
            q_threshold = 0.9 * max(q_values);
            good_actions = q_values > q_threshold;
            n_good = sum(good_actions);
            if n_good > 1
                extra_prob = 0.15 / n_good;
                action_vec(good_actions) = action_vec(good_actions) + extra_prob;
            end
        end
    end
    
    % 确保非负并归一化
    action_vec = max(0, action_vec);
    total = sum(action_vec);
    if total > 1e-10
        action_vec = action_vec / total;
    else
        action_vec = ones(1, obj.action_dim) / obj.action_dim;
    end
    
    % 记录选择
    [~, dominant_action] = max(action_vec);
    obj.recordAction(state_idx, dominant_action);
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
         function prob = softmax(x)
    % 计算softmax概率分布
    exp_x = exp(x - max(x));  % 数值稳定性
    prob = exp_x / sum(exp_x);
end
    end
   
end