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
        
        function action = selectAction(obj, state)
            q_values = obj.Q_table(state, :);
            action = obj.boltzmannAction(state, q_values);
            obj.recordAction(state, action);
        end
        
        function update(obj, state, action, reward, next_state, next_action)
            if isempty(next_action)
                next_action = obj.selectAction(next_state);
            end
            td_error = reward + obj.discount_factor * obj.Q_table(next_state, next_action) ...
                       - obj.Q_table(state, action);
            obj.Q_table(state, action) = obj.Q_table(state, action) + ...
                                       obj.learning_rate * td_error;
            obj.visit_count(state, action) = obj.visit_count(state, action) + 1;
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
