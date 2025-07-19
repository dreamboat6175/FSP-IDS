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
        % TODO: 验证此函数是否被使用
        % TODO: 验证此函数是否被使用
        function obj = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 初始化两个Q表
            obj.Q_table_A = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.Q_table_B = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            
            % 初始化兼容性Q表
            obj.Q_table = (obj.Q_table_A + obj.Q_table_B) / 2;

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
       % 读取ConfigManager中的站点数量
       config = ConfigManager.getDefaultConfig();
       n_stations = config.n_stations;
       
       % 更新兼容性Q表
       obj.updateQTableProperty();
       
       % 健壮性检查
       if isempty(state_vec)
           state_vec = ones(1, obj.state_dim);
       end
       state_vec = reshape(state_vec, 1, []);
       
       % 获取状态索引
       state_idx = obj.encodeState(mean(state_vec));
       state_idx = max(1, min(state_idx, size(obj.Q_table, 1)));
       
       % 获取组合Q值
       Q_combined = obj.Q_table;
       q_values = Q_combined(state_idx, :);
       
       % 确保Q值有效
       if any(isnan(q_values)) || any(isinf(q_values))
           q_values = ones(size(q_values)) * 1.0;
       end
       
       % 动作选择策略
       if rand() < obj.epsilon
           % 探索：随机选择
           action_idx = randi(min(obj.action_dim, n_stations));
       else
           % 利用：选择最佳动作
           [~, action_idx] = max(q_values(1:min(obj.action_dim, n_stations)));
       end
       
       % 生成动作向量 - 长度等于配置中的站点数量
       action_vec = zeros(1, n_stations);
       if action_idx <= n_stations
           action_vec(action_idx) = 1;
       else
           action_vec(1) = 1;  % 默认选择第一个站点
       end
       
       % 归一化确保总和为1（资源分配）
       if sum(action_vec) > 0
           action_vec = action_vec / sum(action_vec);
       else
           action_vec = ones(1, n_stations) / n_stations;  % 均匀分配
       end
       
       % 确保是行向量
       action_vec = reshape(action_vec, 1, []);
       
       % 记录动作（如果基类有这个方法）
       if ismethod(obj, 'recordAction')
           obj.recordAction(state_idx, action_idx);
       end
       
        end

function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
        % 获取状态索引 - 添加边界检查
        state_idx = obj.encodeState(mean(state_vec));
        state_idx = max(1, min(state_idx, size(obj.Q_table_A, 1)));  % 确保在边界内
        
        next_state_idx = obj.encodeState(mean(next_state_vec));
        next_state_idx = max(1, min(next_state_idx, size(obj.Q_table_A, 1)));
        
        % 获取动作索引 - 添加边界检查  
        [~, action_idx] = max(action_vec);
        action_idx = max(1, min(action_idx, size(obj.Q_table_A, 2)));  % 确保在边界内
        
        % 后续的 Q 表访问就安全了
        if rand() < 0.5
            [~, best_action] = max(obj.Q_table_A(next_state_idx, :));
            best_action = max(1, min(best_action, size(obj.Q_table_A, 2)));  % 边界检查
            target = reward + obj.discount_factor * obj.Q_table_B(next_state_idx, best_action);
            td_error = target - obj.Q_table_A(state_idx, action_idx);
            obj.Q_table_A(state_idx, action_idx) = obj.Q_table_A(state_idx, action_idx) + obj.learning_rate * td_error;
        else
            [~, best_action] = max(obj.Q_table_B(next_state_idx, :));
            best_action = max(1, min(best_action, size(obj.Q_table_B, 2)));  % 边界检查
            target = reward + obj.discount_factor * obj.Q_table_A(next_state_idx, best_action);
            td_error = target - obj.Q_table_B(state_idx, action_idx);
            obj.Q_table_B(state_idx, action_idx) = obj.Q_table_B(state_idx, action_idx) + obj.learning_rate * td_error;
        end
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
        end
        
        function load(obj, filename)
            % 加载模型
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
        end
    end
end