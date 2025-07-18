%% DoubleQLearningAgent.m - Double Q-Learning智能体实现（修复版）
% 修复了Q_table属性缺失的问题
classdef DoubleQLearningAgent < RLAgent
    
    properties
        Q_table_A        % 第一个Q值表
        Q_table_B        % 第二个Q值表
        Q_table          % 兼容性Q表（两个Q表的平均）
        visit_count      % 状态-动作访问计数
        lr_scheduler     % 学习率调度器
        strategy_history     % 策略历史记录
        performance_history  % 性能历史记录
        parameter_history    % 参数历史记录
    end
    
    methods
        function obj = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 初始化两个Q表
            obj.Q_table_A = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.Q_table_B = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            
            % 初始化兼容性Q表
            obj.Q_table = (obj.Q_table_A + obj.Q_table_B) / 2;
            
            % 初始化新添加的属性
            % obj.use_softmax = false;     % 默认使用epsilon-greedy
            
            % 确保基类属性有默认值
            obj.strategy_history = [];
            obj.performance_history = struct();
            obj.parameter_history = struct();
            obj.parameter_history.learning_rate = [];
            obj.parameter_history.epsilon = [];
            obj.parameter_history.q_values = [];
        end
        
        function action_vec = selectAction(obj, state_vec)
            % 动作选择
            try
                % 更新兼容性Q表
                obj.updateQTableProperty();
                
                % 健壮性检查
                if isempty(state_vec)
                    state_vec = ones(1, obj.state_dim);
                end
                state_vec = reshape(state_vec, 1, []);
                
                % 获取状态索引
                state_idx = obj.encodeState(mean(state_vec));
                
                % 获取组合Q值
                Q_combined = obj.Q_table;
                q_values = Q_combined(state_idx, :);
                
                % 确保Q值有效
                if any(isnan(q_values)) || any(isinf(q_values))
                    q_values = ones(size(q_values)) * 1.0;
                end
                
                % 动作选择策略
                if obj.use_softmax
                    % Softmax策略
                    temperature = max(0.1, obj.temperature);
                    exp_q = exp(q_values / temperature);
                    action_vec = exp_q / sum(exp_q);
                else
                    % Epsilon-greedy策略
                    if rand() < obj.epsilon
                        % 探索
                        action_vec = rand(1, obj.action_dim);
                    else
                        % 利用
                        action_vec = zeros(1, obj.action_dim);
                        [~, best_action] = max(q_values);
                        action_vec(best_action) = 1;
                    end
                end
                
                % 确保非负并归一化
                action_vec = max(0, action_vec);
                if sum(action_vec) > 0
                    action_vec = action_vec / sum(action_vec);
                else
                    action_vec = ones(1, obj.action_dim) / obj.action_dim;
                end
                
            catch ME
                warning(ME.identifier, 'DoubleQLearningAgent.selectAction 出错: %s', ME.message);
                action_vec = ones(1, obj.action_dim) / obj.action_dim;
            end
            if strcmp(obj.agent_type, 'defender') && length(action_vec) > 1
                obj.strategy_history(end+1, :) = action_vec;
            end
            
            obj.parameter_history.learning_rate(end+1) = obj.learning_rate;
            obj.parameter_history.epsilon(end+1) = obj.epsilon;
            obj.parameter_history.q_values(end+1) = mean((obj.Q_table_A(:) + obj.Q_table_B(:))/2);
        end
        
        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % Double Q-Learning更新
            try
                % 输入验证
                if isempty(action_vec)
                    action_vec = ones(1, 5);
                end
                action_vec = reshape(action_vec, 1, []);
                
                if isempty(state_vec)
                    state_vec = ones(1, obj.state_dim);
                end
                state_vec = reshape(state_vec, 1, []);
                
                if ~isempty(next_state_vec)
                    next_state_vec = reshape(next_state_vec, 1, []);
                else
                    next_state_vec = state_vec;  % 使用当前状态作为默认
                end
                
                % 获取状态索引
                state_idx = obj.encodeState(mean(state_vec));
                next_state_idx = obj.encodeState(mean(next_state_vec));
                
                % 获取动作索引
                [~, action_idx] = max(action_vec);
                action_idx = max(1, min(obj.action_dim, action_idx));
                
                % Double Q-Learning更新
                if rand() < 0.5
                    % 更新Q1，使用Q2来选择动作
                    [~, best_action] = max(obj.Q_table_A(next_state_idx, :));
                    target = reward + obj.discount_factor * obj.Q_table_B(next_state_idx, best_action);
                    td_error = target - obj.Q_table_A(state_idx, action_idx);
                    obj.Q_table_A(state_idx, action_idx) = obj.Q_table_A(state_idx, action_idx) + obj.learning_rate * td_error;
                else
                    % 更新Q2，使用Q1来选择动作
                    [~, best_action] = max(obj.Q_table_B(next_state_idx, :));
                    target = reward + obj.discount_factor * obj.Q_table_A(next_state_idx, best_action);
                    td_error = target - obj.Q_table_B(state_idx, action_idx);
                    obj.Q_table_B(state_idx, action_idx) = obj.Q_table_B(state_idx, action_idx) + obj.learning_rate * td_error;
                end
                
                % 更新访问计数
                obj.visit_count(state_idx, action_idx) = obj.visit_count(state_idx, action_idx) + 1;
                obj.update_count = obj.update_count + 1;
                
                % 更新兼容性Q表
                obj.updateQTableProperty();
                
            catch ME
                warning(ME.identifier, 'DoubleQLearningAgent.update 出错: %s', ME.message);
            end
            obj.recordPerformance(reward, td_error);
        end
        
        function updateQTableProperty(obj)
            % 更新兼容性Q_table属性
            try
                if ~isempty(obj.Q_table_A) && ~isempty(obj.Q_table_B)
                    obj.Q_table = (obj.Q_table_A + obj.Q_table_B) / 2;
                elseif ~isempty(obj.Q_table_A)
                    obj.Q_table = obj.Q_table_A;
                elseif ~isempty(obj.Q_table_B)
                    obj.Q_table = obj.Q_table_B;
                else
                    obj.Q_table = zeros(obj.state_dim, obj.action_dim);
                end
            catch
                obj.Q_table = zeros(obj.state_dim, obj.action_dim);
            end
        end
        
        function policy = getPolicy(obj)
            % 获取当前策略
            try
                % 更新Q_table属性
                obj.updateQTableProperty();
                
                % 检查Q表是否为空
                if isempty(obj.Q_table) || size(obj.Q_table, 1) == 0
                    policy = ones(1, obj.action_dim) / obj.action_dim;
                    return;
                end
                
                % 基于平均Q值的策略
                avg_q_values = mean(obj.Q_table, 1);
                
                if all(avg_q_values == 0) || all(isnan(avg_q_values)) || all(isinf(avg_q_values))
                    policy = ones(1, obj.action_dim) / obj.action_dim;
                    return;
                end
                
                % 使用softmax转换为概率分布
                temperature = 1.0;
                if isprop(obj, 'temperature') || isfield(obj, 'temperature')
                    temperature = max(0.1, obj.temperature);
                end
                
                scaled_q = avg_q_values / temperature;
                exp_q = exp(scaled_q - max(scaled_q));  % 数值稳定性
                policy = exp_q / sum(exp_q);
                
                % 验证策略
                if any(isnan(policy)) || any(isinf(policy)) || sum(policy) == 0
                    policy = ones(1, obj.action_dim) / obj.action_dim;
                else
                    policy = policy / sum(policy);  % 归一化
                end
                
            catch ME
                warning(ME.identifier, 'DoubleQLearningAgent.getPolicy 出错: %s', ME.message);
                policy = ones(1, obj.action_dim) / obj.action_dim;
            end
        end
        
        function stats = getStatistics(obj)
            % 获取统计信息
            try
                % 更新Q_table属性
                obj.updateQTableProperty();
                
                stats = struct();
                stats.name = obj.name;
                stats.agent_type = obj.agent_type;
                stats.update_count = obj.update_count;
                
                % Q表统计
                if ~isempty(obj.Q_table)
                    stats.avg_q_value = mean(obj.Q_table(:));
                    stats.max_q_value = max(obj.Q_table(:));
                    stats.min_q_value = min(obj.Q_table(:));
                    stats.q_value_std = std(obj.Q_table(:));
                else
                    stats.avg_q_value = 0;
                    stats.max_q_value = 0;
                    stats.min_q_value = 0;
                    stats.q_value_std = 0;
                end
                
                % Double Q特有统计
                if ~isempty(obj.Q_table_A) && ~isempty(obj.Q_table_B)
                    stats.q1_avg = mean(obj.Q_table_A(:));
                    stats.q2_avg = mean(obj.Q_table_B(:));
                    stats.q_difference = mean(abs(obj.Q_table_A(:) - obj.Q_table_B(:)));
                end
                
                % 学习参数
                stats.current_learning_rate = obj.learning_rate;
                stats.current_epsilon = obj.epsilon;
                
                % 探索统计
                if ~isempty(obj.visit_count)
                    stats.total_state_visits = sum(obj.visit_count(:));
                    stats.explored_states = sum(sum(obj.visit_count > 0));
                    stats.exploration_ratio = stats.explored_states / numel(obj.visit_count);
                else
                    stats.total_state_visits = 0;
                    stats.explored_states = 0;
                    stats.exploration_ratio = 0;
                end
                
            catch ME
                warning(ME.identifier, 'DoubleQLearningAgent.getStatistics 出错: %s', ME.message);
                stats = struct('name', obj.name, 'agent_type', obj.agent_type, 'update_count', 0);
            end
        end
        
        function state_idx = encodeState(obj, state)
            % 状态编码
            try
                if isempty(state) || ~isnumeric(state)
                    state_idx = 1;
                    return;
                end
                state = double(state(:));
                state(isnan(state)) = 0;
                state(isinf(state)) = 0;
                state_idx = mod(sum(state .* (1:numel(state))'), obj.state_dim) + 1;
                state_idx = max(1, min(obj.state_dim, round(state_idx)));
            catch
                state_idx = 1;
            end
        end
        
        function save(obj, filename)
            % 保存模型
            try
                if nargin < 2
                    filename = sprintf('models/doubleq_%s_%s.mat', ...
                                     obj.agent_type, datestr(now, 'yyyymmdd_HHMMSS'));
                end
                [filepath, ~, ~] = fileparts(filename);
                if ~exist(filepath, 'dir')
                    mkdir(filepath);
                end
                
                save_data.Q_table_A = obj.Q_table_A;
                save_data.Q_table_B = obj.Q_table_B;
                save_data.Q_table = obj.Q_table;
                save_data.visit_count = obj.visit_count;
                save_data.name = obj.name;
                save_data.update_count = obj.update_count;
                
                save(filename, 'save_data');
                fprintf('Double Q-Learning模型已保存: %s\n', filename);
            catch ME
                warning('保存Double Q-Learning模型失败: %s', ME.message);
            end
        end
        
        function load(obj, filename)
            % 加载模型
            try
                if exist(filename, 'file')
                    load_data = load(filename);
                    save_data = load_data.save_data;
                    
                    obj.Q_table_A = save_data.Q_table_A;
                    obj.Q_table_B = save_data.Q_table_B;
                    if isfield(save_data, 'Q_table')
                        obj.Q_table = save_data.Q_table;
                    else
                        obj.updateQTableProperty();
                    end
                    obj.visit_count = save_data.visit_count;
                    obj.name = save_data.name;
                    if isfield(save_data, 'update_count')
                        obj.update_count = save_data.update_count;
                    end
                    
                    fprintf('Double Q-Learning模型已加载: %s\n', filename);
                else
                    error('模型文件不存在: %s', filename);
                end
            catch ME
                warning('加载Double Q-Learning模型失败: %s', ME.message);
            end
        end
    end
end