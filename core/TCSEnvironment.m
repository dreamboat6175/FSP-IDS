classdef TCSEnvironment < handle
    % TCSEnvironment - 基于FSP的列控系统环境
    % 实现理性攻击者（Q-learning）和FSP防御者
    
    properties
        % 系统参数
        n_stations
        n_components
        total_components
        station_values           % 站点价值向量
        
        % 博弈参数
        attacker_strategy       % 当前攻击策略
        defender_strategy       % 当前防守策略
        optimal_defender_strategy
        optimal_attacker_strategy
        
        % FSP相关参数（新增）
        attacker_avg_strategy   % σ̄_A: 防御者感知的攻击者平均策略
        alpha_ewma             % EWMA遗忘因子
        
        % 攻击者Q-learning参数（新增）
        attacker_Q_table       % Q值表
        attacker_lr           % 学习率
        attacker_gamma        % 折扣因子
        attacker_epsilon      % 探索率
        attacker_epsilon_decay
        attacker_epsilon_min
        
        % 防御者RL参数（新增）
        defender_Q_network    % 简化的Q网络
        defender_lr          % 学习率
        defender_gamma       % 折扣因子
        
        % 奖励权重（新增）
        w_radi               % RADI权重
        w_damage             % 损害权重
        
        % 资源参数
        n_resource_types
        n_attack_types
        total_resources
        defense_costs
        defense_effectiveness
        
        % RADI相关
        radi_score           % 当前RADI分数
        radi_history         % RADI历史
        radi_defender        % 兼容性
        radi_attacker        % 兼容性
        
        % 状态和动作空间
        state_dim
        action_dim
        
        % 环境状态
        current_state
        time_step
        
        % 历史记录
        attack_success_rate_history
        attack_target_history
        defense_history
        deployment_history    % 新增：防御部署历史
        damage_history       % 新增：损害历史
        reward_history       % 新增：奖励历史
        attack_history       % 新增：攻击目标历史
        
        % 其他参数
        epsilon
        optimization_method
        
        % 兼容性属性
        attacker_actions
        defender_actions
    end
    
    methods
        function obj = TCSEnvironment(config)
            % 构造函数
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            % 设置资源和攻击类型数量
            if isfield(config, 'n_resource_types')
                obj.n_resource_types = config.n_resource_types;
            else
                if isfield(config, 'resource_types')
                    obj.n_resource_types = length(config.resource_types);
                else
                    obj.n_resource_types = 5;
                end
            end
            
            if isfield(config, 'n_attack_types')
                obj.n_attack_types = config.n_attack_types;
            else
                if isfield(config, 'attack_types')
                    obj.n_attack_types = length(config.attack_types);
                else
                    obj.n_attack_types = 6;
                end
            end
            
            % 设置总资源
            if isfield(config, 'total_resources')
                obj.total_resources = config.total_resources;
            else
                obj.total_resources = 100;
            end
            
            obj.epsilon = 1e-6;
            obj.optimization_method = 'analytical';
            
            % 初始化各个组件
            obj.initializeComponents();
            obj.initializeCosts();
            obj.initializeAttackModel();
            obj.calculateSpaceDimensions();
            
            % 初始化FSP和RL参数
            obj.initializeFSPParameters(config);
            obj.initializeAttackerQLearning(config);
            obj.initializeDefenderRL(config);
            
            % 初始化策略
            obj.initializeStrategies();
            
            % 初始化历史记录
            obj.radi_history = [];
            obj.attack_success_rate_history = [];
            obj.attack_target_history = [];
            obj.defense_history = [];
            obj.deployment_history = [];
            obj.damage_history = [];
            obj.reward_history = struct('attacker', [], 'defender', []);
            obj.attack_history = []; % 新增
            
            obj.reset();
        end
        
        function initializeComponents(obj)
            % 初始化组件和计算站点价值
            component_importance = rand(1, obj.total_components);
            obj.station_values = zeros(1, obj.n_stations);
            
            idx = 1;
            for s = 1:obj.n_stations
                station_components = obj.n_components(s);
                obj.station_values(s) = sum(component_importance(idx:idx+station_components-1));
                idx = idx + station_components;
            end
            
            % 归一化站点价值
            obj.station_values = obj.station_values / sum(obj.station_values);
        end
        
        function initializeFSPParameters(obj, config)
            % 初始化FSP参数
            obj.alpha_ewma = 0.1;  % EWMA遗忘因子
            obj.w_radi = 0.5;      % RADI权重
            obj.w_damage = 0.5;    % 损害权重
            
            if isfield(config, 'fsp_alpha')
                obj.alpha_ewma = config.fsp_alpha;
            end
            if isfield(config, 'w_radi')
                obj.w_radi = config.w_radi;
            end
            if isfield(config, 'w_damage')
                obj.w_damage = config.w_damage;
            end
        end
        
        function initializeAttackerQLearning(obj, config)
            % 初始化攻击者Q-learning参数
            obj.attacker_lr = 0.1;
            obj.attacker_gamma = 0.95;
            obj.attacker_epsilon = 0.3;
            obj.attacker_epsilon_decay = 0.995;
            obj.attacker_epsilon_min = 0.01;
            
            % 初始化Q表（简化版本）
            n_defense_levels = 5;
            state_space_size = n_defense_levels^obj.n_stations;
            obj.attacker_Q_table = zeros(state_space_size, obj.n_stations);
            
            % 从配置读取参数
            if isfield(config, 'attacker_lr')
                obj.attacker_lr = config.attacker_lr;
            end
            if isfield(config, 'attacker_gamma')
                obj.attacker_gamma = config.attacker_gamma;
            end
        end
        
        function initializeDefenderRL(obj, config)
            % 初始化防御者RL参数
            obj.defender_lr = 0.05;
            obj.defender_gamma = 0.95;
            
            % 简化的Q网络
            obj.defender_Q_network = struct();
            obj.defender_Q_network.weights = randn(obj.n_stations, obj.n_stations) * 0.01;
            obj.defender_Q_network.bias = zeros(1, obj.n_stations);
            
            if isfield(config, 'defender_lr')
                obj.defender_lr = config.defender_lr;
            end
        end
        
        function initializeStrategies(obj)
            % 初始化攻防策略
            % 攻击者初始策略：基于站点价值的softmax分布
            temperature = 2.0;
            exp_values = exp(obj.station_values / temperature);
            obj.attacker_strategy = exp_values / sum(exp_values);
            
            % 防御者初始策略：均匀分配
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % FSP：初始化攻击者平均策略为均匀分布
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 初始化最优策略
            obj.optimal_attacker_strategy = obj.attacker_strategy;
            obj.optimal_defender_strategy = obj.defender_strategy;
        end
        
        function state = reset(obj)
            % 重置环境
            obj.time_step = 0;
            
            % 重置策略
            obj.initializeStrategies();
            
            % 重置Q表
            obj.attacker_Q_table(:) = 0;
            
            % 重置历史记录
            obj.radi_history = [];
            obj.attack_success_rate_history = [];
            obj.attack_target_history = [];
            obj.defense_history = [];
            obj.deployment_history = [];
            obj.damage_history = [];
            obj.reward_history = struct('attacker', [], 'defender', []);
            obj.attack_history = []; % 新增
            
            % 重置RADI
            obj.radi_score = 0;
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            
            state = obj.generateState();
            obj.current_state = state;
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_action, attacker_action)
            % 执行一步环境交互 - 基于新模型
            
            % 1. 解析动作
            defender_deployment = obj.parseDefenderAction(defender_action);
            attacker_target = obj.parseAttackerAction(attacker_action);
            
            % 2. FSP: 更新攻击者平均策略（EWMA）
            h_A = zeros(1, obj.n_stations);
            h_A(attacker_target) = 1;
            obj.attacker_avg_strategy = (1 - obj.alpha_ewma) * obj.attacker_avg_strategy + ...
                                       obj.alpha_ewma * h_A;
            obj.attacker_avg_strategy = obj.attacker_avg_strategy / sum(obj.attacker_avg_strategy);
            
            % 3. 计算攻击结果
            defense_at_target = defender_deployment(attacker_target);
            success_rate = 1 - tanh(defense_at_target / obj.total_resources);
            attack_success = rand() < success_rate;
            
            % 4. 计算奖励
            % 攻击者奖励：R_A = Success(A_A, d_t) × Value(A_A)
            if attack_success
                reward_att = success_rate * obj.station_values(attacker_target);
                damage = obj.station_values(attacker_target);
            else
                reward_att = 0;
                damage = 0;
            end
            
            % 防御者奖励：复合奖励函数
            % 计算事后最优部署
            optimal_deployment = obj.computeOptimalDeploymentAfterAttack(attacker_target);
            
            % 计算RADI
            obj.radi_score = obj.calculateRADI(defender_deployment, optimal_deployment);
            
            % 防御者奖励：R_D = w_radi(1-RADI) + w_damage(1-Damage)
            reward_def = obj.w_radi * (1 - obj.radi_score) + obj.w_damage * (1 - damage);
            
            % 5. 更新Q-learning
            obj.updateAttackerQ(defender_deployment, attacker_target, reward_att);
            obj.updateDefenderQ(obj.attacker_avg_strategy, defender_deployment, reward_def);
            
            % 6. 更新历史记录
            obj.updateHistory(attack_success, damage, attacker_target, defender_deployment);
            
            % 7. 生成下一状态
            next_state = obj.generateState();
            obj.current_state = next_state;
            obj.time_step = obj.time_step + 1;
            
            % 8. 准备信息
            info = obj.prepareStepInfo(attack_success, attacker_target, defender_deployment, damage);
        end
        
        function optimal = computeOptimalDeploymentAfterAttack(obj, actual_attack)
            % 计算事后最优防御部署（知道攻击目标后）
            optimal = zeros(1, obj.n_stations);
            
            % 70%资源给被攻击站点
            main_allocation = 0.7;
            optimal(actual_attack) = obj.total_resources * main_allocation;
            
            % 30%根据站点价值分配给其他站点
            remaining_resources = obj.total_resources * (1 - main_allocation);
            other_stations = setdiff(1:obj.n_stations, actual_attack);
            
            if ~isempty(other_stations)
                values = obj.station_values(other_stations);
                values = values / sum(values);
                for i = 1:length(other_stations)
                    optimal(other_stations(i)) = remaining_resources * values(i);
                end
            end
        end
        
        function radi = calculateRADI(obj, actual, optimal)
            % 计算RADI - 使用相对偏差
            radi = 0;
            valid_count = 0;
            
            for i = 1:length(actual)
                if optimal(i) > obj.epsilon
                    relative_deviation = ((actual(i) - optimal(i))^2) / (optimal(i)^2);
                    radi = radi + relative_deviation;
                    valid_count = valid_count + 1;
                end
            end
            
            if valid_count > 0
                radi = radi / valid_count;
            end
            
            % 限制范围
            radi = min(radi, 10);
        end
        
        function updateAttackerQ(obj, defender_state, action, reward)
            % 更新攻击者Q表
            state_idx = obj.encodeDefenderState(defender_state);
            
            % Q-learning更新
            current_q = obj.attacker_Q_table(state_idx, action);
            max_next_q = max(obj.attacker_Q_table(state_idx, :));
            
            td_target = reward + obj.attacker_gamma * max_next_q;
            td_error = td_target - current_q;
            
            obj.attacker_Q_table(state_idx, action) = current_q + obj.attacker_lr * td_error;
            
            % 更新探索率
            obj.attacker_epsilon = max(obj.attacker_epsilon_min, ...
                                      obj.attacker_epsilon * obj.attacker_epsilon_decay);
        end
        
        function updateDefenderQ(obj, state, action, reward)
            % 更新防御者Q网络
            [~, action_idx] = max(action);
            
            % 计算Q值
            q_values = state * obj.defender_Q_network.weights + obj.defender_Q_network.bias;
            
            % 简化的梯度更新
            gradient = state' * reward;
            obj.defender_Q_network.weights(:, action_idx) = ...
                obj.defender_Q_network.weights(:, action_idx) + obj.defender_lr * gradient;
        end
        
        function state_idx = encodeDefenderState(obj, deployment)
            % 编码防守部署为离散状态索引
            n_levels = 5;
            levels = zeros(1, obj.n_stations);
            
            normalized_deployment = deployment / obj.total_resources;
            for i = 1:obj.n_stations
                levels(i) = min(n_levels-1, floor(normalized_deployment(i) * n_levels));
            end
            
            % 转换为索引
            state_idx = 1;
            multiplier = 1;
            for i = 1:obj.n_stations
                state_idx = state_idx + levels(i) * multiplier;
                multiplier = multiplier * n_levels;
            end
        end
        
        function target = selectAttackerAction(obj, defender_deployment)
            % 理性攻击者选择动作（ε-贪婪）
            if rand() < obj.attacker_epsilon
                % 探索：随机选择
                target = randi(obj.n_stations);
            else
                % 利用：基于Q值选择
                state_idx = obj.encodeDefenderState(defender_deployment);
                [~, target] = max(obj.attacker_Q_table(state_idx, :));
            end
        end
        
        function deployment = computeDefenderBestResponse(obj)
            % FSP防御者：基于攻击者平均策略计算最佳响应
            
            % 使用Q网络计算Q值
            q_values = obj.attacker_avg_strategy * obj.defender_Q_network.weights + ...
                      obj.defender_Q_network.bias;
            
            % Softmax转换为概率
            temperature = 0.5;
            exp_q = exp(q_values / temperature);
            deployment_probs = exp_q / sum(exp_q);
            
            % 分配资源
            deployment = deployment_probs * obj.total_resources;
        end
        
        function updateHistory(obj, attack_success, damage, attacker_target, defender_deployment)
            % 更新历史记录
            
            % 计算并记录攻击成功率
            if attack_success
                obj.attack_success_rate_history(end+1) = 1;
            else
                obj.attack_success_rate_history(end+1) = 0;
            end
            
            % 记录攻击目标
            attack_vector = zeros(1, obj.n_stations);
            attack_vector(attacker_target) = 1;
            if isempty(obj.attack_target_history)
                obj.attack_target_history = attack_vector;
            else
                obj.attack_target_history(end+1, :) = attack_vector;
            end
            
            % 记录防御部署
            obj.defense_history(end+1, :) = defender_deployment;
            obj.deployment_history(end+1, :) = defender_deployment;
            
            % 记录损害和RADI
            obj.damage_history(end+1) = damage;
            obj.radi_history(end+1) = obj.radi_score;
            
            % 更新RADI（兼容性）
            obj.radi_defender = obj.radi_score;
            obj.radi_attacker = 0;  % 简化处理
            % 新增：记录攻击目标编号
            obj.attack_history(end+1) = attacker_target;
        end
        
        function info = prepareStepInfo(obj, attack_success, attacker_target, defender_deployment, damage)
            % 准备步骤信息
            info = struct();
            
            info.attack_success = attack_success;
            info.attack_target = attacker_target;
            info.defender_deployment = defender_deployment;
            info.radi_score = obj.radi_score;
            info.damage = damage;
            info.attacker_avg_strategy = obj.attacker_avg_strategy;
            info.time_step = obj.time_step;
            
            % 兼容性字段
            info.resource_allocation = defender_deployment / sum(defender_deployment);
            info.current_strategies = struct();
            info.current_strategies.attacker = obj.attacker_strategy;
            info.current_strategies.defender = obj.defender_strategy;
        end
        
        % ========== 兼容性方法 ==========
        
        function allocation = parseDefenderAction(obj, defender_action)
            % 解析防御者动作
            if length(defender_action) == obj.n_stations
                allocation = defender_action;
            else
                allocation = zeros(1, obj.n_stations);
                if defender_action >= 1 && defender_action <= obj.n_stations
                    allocation(defender_action) = obj.total_resources;
                end
            end
            
            % 确保资源约束
            if sum(allocation) > obj.total_resources
                allocation = allocation / sum(allocation) * obj.total_resources;
            end
        end
        
        function target = parseAttackerAction(obj, attacker_action)
            % 解析攻击者动作
            if attacker_action >= 1 && attacker_action <= obj.n_stations
                target = attacker_action;
            else
                target = randi(obj.n_stations);
            end
        end
        
        function state = generateState(obj)
            % 生成状态向量
            state = zeros(1, obj.state_dim);
            
            % 当前防御部署
            idx = 1;
            if ~isempty(obj.defense_history)
                current_deployment = obj.defense_history(end, :) / obj.total_resources;
                state(idx:idx+obj.n_stations-1) = current_deployment;
            else
                state(idx:idx+obj.n_stations-1) = obj.defender_strategy;
            end
            idx = idx + obj.n_stations;
            
            % 攻击者平均策略
            state(idx:idx+obj.n_stations-1) = obj.attacker_avg_strategy;
            idx = idx + obj.n_stations;
            
            % 站点价值
            state(idx:idx+obj.n_stations-1) = obj.station_values;
            idx = idx + obj.n_stations;
            
            % 最近的成功率
            if length(obj.attack_success_rate_history) > 10
                state(idx) = mean(obj.attack_success_rate_history(end-9:end));
            else
                state(idx) = 0.5;
            end
        end
        
        function calculateSpaceDimensions(obj)
            % 计算状态和动作空间维度
            obj.state_dim = obj.n_stations * 3 + 1;  % 防御部署 + 攻击策略 + 站点价值 + 成功率
            obj.action_dim = obj.n_stations;
        end
        
        function initializeCosts(obj)
            % 初始化成本
            obj.defense_costs = zeros(obj.n_stations, obj.n_resource_types);
            for i = 1:obj.n_stations
                for j = 1:obj.n_resource_types
                    base_cost = 10 + j * 5;
                    obj.defense_costs(i, j) = base_cost * (1 + obj.station_values(i));
                end
            end
            
            % 初始化防御效果
            obj.defense_effectiveness = ones(1, obj.n_resource_types) * 0.7;
        end
        
        function initializeAttackModel(obj)
            % 初始化攻击模型（简化）
            % 实际实现中应包含更详细的攻击类型建模
        end
        
        % 保留原有的兼容性方法
        function optimal_defense = computeOptimalDefenseStrategy(obj, attack_strategy)
            % 基于当前攻击策略计算最优防守（用于RADI计算）
            % 这是防御者不知道具体攻击目标时的最优策略
            
            % 基于威胁评估分配资源
            threat_assessment = attack_strategy .* obj.station_values;
            threat_assessment = threat_assessment / sum(threat_assessment);
            
            optimal_defense = threat_assessment;
        end
        
        function radi = calculateRADIScore(obj, current_strategy, optimal_strategy)
            % 计算RADI分数（兼容性方法）
            radi = obj.calculateRADI(current_strategy, optimal_strategy);
        end
        
        % 其他必要的兼容性方法...
    end
end