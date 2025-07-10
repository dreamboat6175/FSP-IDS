%% SARSAAgent.m - SARSA智能体实现
classdef SARSAAgent < RLAgent
    properties
        Q_table
        visit_count
    end
    
    methods
        function obj = SARSAAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.Q_table = zeros(state_dim, action_dim);
            obj.visit_count = zeros(state_dim, action_dim);
            obj.Q_table = obj.Q_table + randn(state_dim, action_dim) * 0.01;
            obj.use_softmax = true;
        end

        function action = selectAction(obj, state_vec)
            % ===== 修改开始 =====
            % 1. 把状态清单（state_vec）翻译成页码（state_idx）
            state_idx = obj.getStateIndex(state_vec);
            
            % 2. 使用页码查询Q表
            q_values = obj.Q_table(state_idx, :);
            
            if obj.use_softmax
                action = obj.boltzmannAction(state_idx, q_values);
            else
                action = obj.epsilonGreedyAction(state_idx, q_values);
            end
            
            obj.recordAction(state_idx, action);
            % ===== 修改结束 =====
        end

        function update(obj, state_vec, action, reward, next_state_vec, next_action)
            % 将状态向量转换为索引
            state_idx = obj.getStateIndex(state_vec);
            next_state_idx = obj.getStateIndex(next_state_vec);
            
            % 如果没有提供下一个动作，则需要根据下一个状态选择一个
            if isempty(next_action)
                next_action = obj.selectAction(next_state_vec);
            end
            
            % 使用索引来更新Q表
            td_error = reward + obj.discount_factor * obj.Q_table(next_state_idx, next_action) ...
                       - obj.Q_table(state_idx, action);
            obj.Q_table(state_idx, action) = obj.Q_table(state_idx, action) + ...
                                       obj.learning_rate * td_error;
                                       
            obj.visit_count(state_idx, action) = obj.visit_count(state_idx, action) + 1;
            obj.recordReward(reward);
            obj.update_count = obj.update_count + 1;
        end
        
        function policy = getPolicy(obj)
            policy = obj.Q_table;
        end
        
        function save(obj, filename)
            if nargin < 2
                filename = sprintf('models/sarsa_%s_%s.mat', ...
                                 obj.agent_type, datestr(now, 'yyyymmdd_HHMMSS'));
            end
            [filepath, ~, ~] = fileparts(filename);
            if ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            save_data.Q_table = obj.Q_table;
            save_data.visit_count = obj.visit_count;
            save_data.name = obj.name;
            save_data.agent_type = obj.agent_type;
            save_data.update_count = obj.update_count;
            save(filename, 'save_data');
        end
        
        function load(obj, filename)
            if exist(filename, 'file')
                load_data = load(filename);
                save_data = load_data.save_data;
                obj.Q_table = save_data.Q_table;
                obj.visit_count = save_data.visit_count;
                obj.name = save_data.name;
                obj.update_count = save_data.update_count;
            else
                error('模型文件不存在: %s', filename);
            end
        end
    end
end
