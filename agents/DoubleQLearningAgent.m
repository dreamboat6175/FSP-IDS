%% DoubleQLearningAgent.m - Double Q-Learning智能体实现
classdef DoubleQLearningAgent < RLAgent
    properties
        Q1_table
        Q2_table
        visit_count
    end
    
    methods
        function obj = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.Q1_table = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.Q2_table = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.visit_count = zeros(state_dim, action_dim);
        end
        
        function action = selectAction(obj, state_vec)
            % ===== 修正开始 =====
            % 1. 将状态向量转换为索引
            state_idx = obj.getStateIndex(state_vec);
            
            % 2. 使用两个Q表的平均值进行决策
            Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
            q_values = Q_combined(state_idx, :);
            
            % 3. 使用ε-贪婪策略选择动作
            action = obj.epsilonGreedyAction(state_idx, q_values);
            obj.recordAction(state_idx, action);
            % ===== 修正结束 =====
        end
        
        function update(obj, state_vec, action, reward, next_state_vec, ~)
            % 1. 将状态向量转换为索引
            state_idx = obj.getStateIndex(state_vec);
            next_state_idx = obj.getStateIndex(next_state_vec);

            % 2. 使用索引进行Double Q-Learning更新
            if rand() < 0.5
                [~, best_action] = max(obj.Q1_table(next_state_idx, :));
                td_error = reward + obj.discount_factor * obj.Q2_table(next_state_idx, best_action) ...
                          - obj.Q1_table(state_idx, action);
                obj.Q1_table(state_idx, action) = obj.Q1_table(state_idx, action) + ...
                                            obj.learning_rate * td_error;
            else
                [~, best_action] = max(obj.Q2_table(next_state_idx, :));
                td_error = reward + obj.discount_factor * obj.Q1_table(next_state_idx, best_action) ...
                          - obj.Q2_table(state_idx, action);
                obj.Q2_table(state_idx, action) = obj.Q2_table(state_idx, action) + ...
                                            obj.learning_rate * td_error;
            end
            obj.visit_count(state_idx, action) = obj.visit_count(state_idx, action) + 1;
            obj.recordReward(reward);
            obj.update_count = obj.update_count + 1;
        end
        
        function policy = getPolicy(obj)
            policy = (obj.Q1_table + obj.Q2_table) / 2;
        end
        
        function save(obj, filename)
            if nargin < 2
                filename = sprintf('models/doubleq_%s_%s.mat', ...
                                 obj.agent_type, datestr(now, 'yyyymmdd_HHMMSS'));
            end
            [filepath, ~, ~] = fileparts(filename);
            if ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            save_data.Q1_table = obj.Q1_table;
            save_data.Q2_table = obj.Q2_table;
            save_data.visit_count = obj.visit_count;
            save_data.name = obj.name;
            save(filename, 'save_data');
        end
        
        function load(obj, filename)
            if exist(filename, 'file')
                load_data = load(filename);
                save_data = load_data.save_data;
                obj.Q1_table = save_data.Q1_table;
                obj.Q2_table = save_data.Q2_table;
                obj.visit_count = save_data.visit_count;
                obj.name = save_data.name;
            else
                error('模型文件不存在: %s', filename);
            end
        end
    end
end