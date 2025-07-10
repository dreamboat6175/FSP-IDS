%% QLearningAgent.m - Q-Learning智能体实现
% =========================================================================
% 描述: 实现标准Q-Learning算法的智能体
% =========================================================================

classdef QLearningAgent < RLAgent
    
    properties
        Q_table    % Q值表
        visit_count % 状态-动作访问计数
    end
    
    methods
        function obj = QLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 初始化Q表
            obj.Q_table = zeros(state_dim, action_dim);
            obj.visit_count = zeros(state_dim, action_dim);
            
            % 优化的初始化：使用较小的正值鼓励探索
            obj.Q_table = obj.Q_table + 0.1 + randn(state_dim, action_dim) * 0.05;
        end

        function action = selectAction(obj, state_vec)
            state_idx = obj.getStateIndex(state_vec); % 将向量状态转换为索引
            q_values = obj.Q_table(state_idx, :);
            
            if obj.use_softmax
                action = obj.boltzmannAction(state_idx, q_values);
            else
                action = obj.epsilonGreedyAction(state_idx, q_values);
            end
            obj.recordAction(state_idx, action);
        end
        
        function update(obj, state_vec, action, reward, next_state_vec, ~)
            % Q-Learning更新规则
            
            % ===== 修改开始 =====
            % 1. 把当前和下一个状态清单（向量）都翻译成页码（索引）
            state_idx = obj.getStateIndex(state_vec);
            next_state_idx = obj.getStateIndex(next_state_vec);
            
            % 2. 使用页码（索引）进行后续计算
            max_next_q = max(obj.Q_table(next_state_idx, :));
            
            % 计算TD误差
            td_error = reward + obj.discount_factor * max_next_q - obj.Q_table(state_idx, action);
            
            % 更新Q值
            obj.Q_table(state_idx, action) = obj.Q_table(state_idx, action) + ...
                                       obj.learning_rate * td_error;
            
            % 更新访问计数
            obj.visit_count(state_idx, action) = obj.visit_count(state_idx, action) + 1;
            
            % 记录奖励和更新计数
            obj.recordReward(reward);
            obj.update_count = obj.update_count + 1;
            % ===== 修改结束 =====
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