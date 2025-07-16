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
            % 健壮性检查：确保n_components_per_station长度正确
            if isfield(config, 'n_components_per_station')
                obj.n_components = config.n_components_per_station(:)'; % 保证是行向量
                if length(obj.n_components) ~= obj.n_stations
                    error('n_components_per_station 长度必须等于 n_stations');
                end
            else
                % 默认每站点3个组件
                obj.n_components = ones(1, obj.n_stations) * 3;
            end
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
            
            % 修正：Q表状态数与encodeDefenderState一致，防止内存爆炸
            n_defense_states = 50; % 必须与encodeDefenderState一致
            obj.attacker_Q_table = zeros(n_defense_states, obj.n_stations);
            obj.attacker_Q_table = obj.attacker_Q_table + 0.1;
            obj.attacker_epsilon = 0.8;
            obj.attacker_epsilon_decay = 0.999;
            obj.attacker_epsilon_min = 0.05;
            
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
    
    % 防御者初始策略：只在第一次初始化时设为均匀分配
    if isempty(obj.defender_strategy)
        obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
    end
    % 后续reset时保持之前学到的策略
    
    % FSP：初始化攻击者平均策略为均匀分布
    if isempty(obj.attacker_avg_strategy)
        obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
    end
    
    % 初始化最优策略
    obj.optimal_attacker_strategy = obj.attacker_strategy;
    obj.optimal_defender_strategy = obj.defender_strategy;
end
        
        function state = reset(obj)
    % 重置环境状态但保持学习进度
    obj.time_step = 0;
    
    % 修正：只在第一次调用时初始化策略
    if isempty(obj.attacker_strategy)
        obj.initializeStrategies();
    end
    
    % 修正：不要清空Q表！这是关键问题
    % obj.attacker_Q_table(:) = 0; % 删除这行
    
    % 重置单回合历史记录
    obj.radi_history = [];
    obj.attack_success_rate_history = [];
    obj.attack_target_history = [];
    obj.defense_history = [];
    
    % 重置RADI
    obj.radi_score = 0;
    
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
    % 使用配置中的alpha值（确保已设置）
    if ~isfield(obj, 'alpha_ewma') || obj.alpha_ewma == 0
        obj.alpha_ewma = 0.2;  % 默认值
    end
    obj.attacker_avg_strategy = (1 - obj.alpha_ewma) * obj.attacker_avg_strategy + ...
                               obj.alpha_ewma * h_A;
    obj.attacker_avg_strategy = obj.attacker_avg_strategy / sum(obj.attacker_avg_strategy);
    
    % 3. 计算攻击结果
    defense_at_target = defender_deployment(attacker_target);
    success_rate = 1 - tanh(defense_at_target / obj.total_resources);
    attack_success = rand() < success_rate;
    
    % 4. 计算损害
    if attack_success
        damage = obj.station_values(attacker_target) * (1 - defense_at_target/obj.total_resources);
    else
        damage = 0;
    end
    
    % 5. 计算奖励（调用新方法）
    [reward_def, reward_att] = obj.calculateRewards(attack_success, damage, attacker_target, defender_deployment);
    
    % 6. 更新Q-learning和FSP策略
    obj.updateAttackerQ(defender_deployment, attacker_target, reward_att);
    obj.updateDefenderQ(obj.attacker_avg_strategy, defender_deployment, reward_def);
    
    % 7. 更新历史记录
    obj.updateHistory(attack_success, damage, attacker_target, defender_deployment);
    
    % 8. 生成下一状态
    next_state = obj.generateState();
    obj.current_state = next_state;
    obj.time_step = obj.time_step + 1;
    
    % 9. 准备信息
    info = obj.prepareStepInfo(attack_success, attacker_target, defender_deployment, damage);
end
        
        function optimal = computeOptimalDeploymentAfterAttack(obj, actual_attack)
    % 计算事后最优防御部署
    optimal = zeros(1, obj.n_stations);
    
    % 修正：更合理的资源分配策略
    % 50%给被攻击站点，50%根据威胁和价值分配
    main_allocation = 0.5;
    optimal(actual_attack) = obj.total_resources * main_allocation;
    
    % 剩余资源根据站点价值和威胁分配
    remaining_resources = obj.total_resources * (1 - main_allocation);
    other_stations = setdiff(1:obj.n_stations, actual_attack);
    
    if ~isempty(other_stations)
        % 结合站点价值和感知威胁
        values = obj.station_values(other_stations);
        threats = obj.attacker_avg_strategy(other_stations);
        combined_weights = values .* (1 + threats); % 价值×威胁权重
        combined_weights = combined_weights / sum(combined_weights);
        
        for i = 1:length(other_stations)
            optimal(other_stations(i)) = remaining_resources * combined_weights(i);
        end
    end
end
        
        function radi = calculateRADI(obj, actual, optimal)
    % 计算RADI - 使用标准化的相对误差
    
    % 避免除零错误
    epsilon = 1e-6;
    optimal = optimal + epsilon;
    
    % 计算相对误差
    relative_errors = abs(actual - optimal) ./ optimal;
    
    % 使用加权平均，权重为站点价值
    weights = obj.station_values / sum(obj.station_values);
    radi = sum(weights .* relative_errors);
    
    % 修正：限制RADI在合理范围内
    radi = min(radi, 5.0); % 降低上限
end
        
        function updateAttackerQ(obj, defender_state, action, reward)
    % 更新攻击者Q表
    state_idx = obj.encodeDefenderState(defender_state);
    
    % 修正：改进奖励设计
    % 考虑攻击成功的收益和防御强度的惩罚
    defense_strength = defender_state(action) / obj.total_resources;
    adjusted_reward = reward - defense_strength * 0.5; % 防御强度惩罚
    
    % Q-learning更新
    current_q = obj.attacker_Q_table(state_idx, action);
    
    % 计算下一状态的最大Q值（简化为当前状态）
    max_next_q = max(obj.attacker_Q_table(state_idx, :));
    
    td_target = adjusted_reward + obj.attacker_gamma * max_next_q;
    td_error = td_target - current_q;
    
    obj.attacker_Q_table(state_idx, action) = current_q + obj.attacker_lr * td_error;
    
    % 更新探索率
    obj.attacker_epsilon = max(obj.attacker_epsilon_min, ...
                              obj.attacker_epsilon * obj.attacker_epsilon_decay);
end
        
        function updateDefenderQ(obj, state, action, reward)
    % FSP防御者策略更新
    % 使用历史最佳响应的平均作为最终策略
    
    % 记录当前最佳响应
    current_best_response = action / sum(action); % 归一化
    
    % 使用EWMA更新防御者平均策略
    beta = 0.05; % 防御者策略更新速率
    if isempty(obj.defender_strategy) || all(obj.defender_strategy == 1/obj.n_stations)
        obj.defender_strategy = current_best_response;
    else
        obj.defender_strategy = (1 - beta) * obj.defender_strategy + beta * current_best_response;
    end
    
    % 确保归一化
    obj.defender_strategy = obj.defender_strategy / sum(obj.defender_strategy);
end
        
       function state_idx = encodeDefenderState(obj, deployment)
    % 简化的状态编码
    n_levels = 3;  % 减少到3个级别
    n_defense_states = 50;  % 总状态数
    
    % 归一化部署
    normalized = deployment / obj.total_resources;
    
    % 找出资源最多的站点
    [~, max_idx] = max(normalized);
    
    % 计算资源集中度
    concentration = max(normalized);
    concentration_level = min(floor(concentration * n_levels), n_levels-1);
    
    % 编码为索引
    state_idx = (max_idx - 1) * n_levels + concentration_level + 1;
    state_idx = min(state_idx, n_defense_states);
end
        
        function target = selectAttackerAction(obj, defender_deployment)
    % 理性攻击者选择动作（ε-贪婪）
    if rand() < obj.attacker_epsilon
        % 探索：基于站点价值的加权随机选择
        value_probs = obj.station_values / sum(obj.station_values);
        % 替换randsample为rand+cumsum实现
        edges = [0, cumsum(value_probs(:)')];
        r = rand();
        target = find(r >= edges(1:end-1) & r < edges(2:end), 1, 'first');
    else
        % 利用：基于Q值选择
        state_idx = obj.encodeDefenderState(defender_deployment);
        % 获取Q值并添加一些噪声以打破平局
        q_values = obj.attacker_Q_table(state_idx, :);
        q_values = q_values + randn(size(q_values)) * 0.01;
        % 考虑站点价值的加权Q值
        weighted_q = q_values .* obj.station_values;
        [~, target] = max(weighted_q);
    end
end
        
        function updateAttackerStrategy(obj)
    % 基于Q表更新攻击者的混合策略
    % 计算所有状态下的平均Q值
    avg_q_values = mean(obj.attacker_Q_table, 1);
    
    % 使用softmax将Q值转换为概率分布
    temperature = 1.0;
    exp_q = exp(avg_q_values / temperature);
    obj.attacker_strategy = exp_q / sum(exp_q);
    
    % 确保策略有效
    obj.attacker_strategy = max(obj.attacker_strategy, 1e-6);
    obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
end
        
        function deployment = computeDefenderBestResponse(obj)
    % FSP防御者：基于攻击者平均策略计算最佳响应
    
    % 获取感知的攻击威胁
    threat_probs = obj.attacker_avg_strategy;
    
    % 修正：使用博弈论最佳响应
    % 期望损失 = 攻击概率 × 站点价值 × (1 - 防御效果)
    expected_losses = threat_probs .* obj.station_values;
    
    % 基于期望损失分配资源
    if sum(expected_losses) > 0
        threat_weights = expected_losses / sum(expected_losses);
    else
        threat_weights = ones(1, obj.n_stations) / obj.n_stations;
    end
    
    % 80%基于威胁分配，20%均匀分配（保持基础防护）
    threat_based = 0.8;
    uniform_based = 0.2;
    
    deployment = threat_based * threat_weights * obj.total_resources + ...
                uniform_based * ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
    
    % 修正：更新防御者策略
    obj.defender_strategy = deployment / obj.total_resources;
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
        
        function [reward_def, reward_att] = calculateRewards(obj, attack_success, damage, attacker_target, defender_deployment)
            % 计算攻击者奖励
            if attack_success
                % 成功攻击：获得与站点价值成比例的奖励
                base_reward = obj.station_values(attacker_target);
                % 考虑防御强度的影响
                defense_factor = defender_deployment(attacker_target) / obj.total_resources;
                reward_att = base_reward * (1 - defense_factor * 0.5);
            else
                % 攻击失败：小惩罚
                reward_att = -0.1;
            end
            % 计算防御者奖励（复合奖励函数）
            optimal_deployment = obj.computeOptimalDeploymentAfterAttack(attacker_target);
            obj.radi_score = obj.calculateRADI(defender_deployment, optimal_deployment);
            % 修正：改进奖励函数
            reward_radi = obj.w_radi * exp(-obj.radi_score); % 指数衰减，RADI越小奖励越大
            reward_damage = obj.w_damage * (1 - damage / max(obj.station_values)); % 标准化损害
            reward_def = reward_radi + reward_damage;
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
            if isscalar(attacker_action) && attacker_action >= 1 && attacker_action <= obj.n_stations
                target = attacker_action;
            else
                target = randi(obj.n_stations);
            end
        end
        
        function state = generateState(obj)
            % 生成简化的状态向量
            state = zeros(1, obj.state_dim);
            
            % 使用当前防御部署的前几个元素作为状态
            if ~isempty(obj.defense_history)
                current_deployment = obj.defense_history(end, :) / obj.total_resources;
            else
                current_deployment = obj.defender_strategy;
            end
            
            % 只使用前几个元素，避免状态空间过大
            n_elements = min(length(current_deployment), obj.state_dim);
            state(1:n_elements) = current_deployment(1:n_elements);
        end
        
        function calculateSpaceDimensions(obj)
            % 计算状态和动作空间维度
            % 简化状态表示：只使用站点数量作为状态维度
            obj.state_dim = min(50, obj.n_stations * 2);  % 限制最大状态数为50
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