%% AgentFactory.m - 智能体工厂类 (改进版)
% =========================================================================
% 描述: 负责创建和配置各种类型的强化学习智能体
% 改进版本：支持新的配置结构和探索策略
% =========================================================================

classdef AgentFactory
    
    methods (Static)
        function agents = createDefenderAgents(config, environment)
            % 创建防御者智能体数组（兼容旧接口）
            
            % 获取算法列表
            if isfield(config, 'algorithms')
                algorithms = config.algorithms;
            else
                algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            end
            
            % 获取维度
            state_dim = environment.state_dim;
            action_dim = environment.action_dim_defender;
            
            % 使用新的批量创建方法
            agents = AgentFactory.createMultipleAgents(algorithms, 'defender', config, state_dim, action_dim);
        end
        
        function agent = createAttackerAgent(config, environment)
            % 创建攻击者智能体（兼容旧接口）
            
            % 获取算法
            if isfield(config, 'attacker_algorithm')
                algorithm = config.attacker_algorithm;
            else
                algorithm = 'QLearning';
            end
            
            % 获取维度
            state_dim = environment.state_dim;
            action_dim = environment.action_dim_attacker;
            
            % 使用新的创建方法
            agent = AgentFactory.createAgent(algorithm, 'Attacker', 'attacker', config, state_dim, action_dim);
        end
        
        function agent = createAgent(algorithm, name, agent_type, config, state_dim, action_dim)
            % 创建智能体
            % 输入:
            %   algorithm - 算法类型 ('QLearning', 'SARSA', 'DoubleQLearning', 'DQN')
            %   name - 智能体名称
            %   agent_type - 智能体类型 ('defender' 或 'attacker')
            %   config - 配置参数
            %   state_dim - 状态空间维度
            %   action_dim - 动作空间维度
            
            % --- This comment serves no functional purpose but helps refresh MATLAB's class definition. ---
            
            % 确保配置完整性
            config = AgentFactory.ensureConfigCompleteness(config);
            
            % 根据智能体类型调整配置
            if strcmpi(agent_type, 'attacker')
                config = AgentFactory.configureAttackerParams(config);
            end
            
            % 创建智能体
            switch upper(algorithm)
                case 'QLEARNING'
                    agent = QLearningAgent(name, agent_type, config, state_dim, action_dim);
                    
                case 'SARSA'
                    agent = SARSAAgent(name, agent_type, config, state_dim, action_dim);
                    
                case 'DOUBLEQLEARNING'
                    agent = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim);
                    
                case 'DQN' % 明确处理 DQN 算法
                    try
                        % 尝试创建 DQNAgent。如果 DQNAgent.m 文件不存在，这将抛出错误。
                        agent = DQNAgent(name, agent_type, config, state_dim, action_dim);
                        fprintf('✓ 创建 %s 智能体: %s (DQN)\n', agent_type, name);
                    catch ME
                        % 如果 DQNAgent 类未找到或定义不正确，则回退到 QLearningAgent
                        if strcmp(ME.identifier, 'MATLAB:UndefinedFunction') || strcmp(ME.identifier, 'MATLAB:class:UndefinedClass')
                            warning('AgentFactory:DQNAgentNotFound', ...
                                    'DQN 智能体类 (DQNAgent.m) 未找到或定义不正确。将使用 QLearningAgent 作为替代。');
                            agent = QLearningAgent(name, agent_type, config, state_dim, action_dim); % 回退
                        else
                            rethrow(ME); % 重新抛出其他类型的错误
                        end
                    end
                    
                otherwise
                    error('AgentFactory:UnknownAlgorithm', ...
                          '未知的算法类型: %s', algorithm);
            end
            
            % 只有当智能体不是因错误而回退时才打印此行，或者在回退逻辑中打印
            if ~exist('ME', 'var') || ~(strcmp(ME.identifier, 'MATLAB:UndefinedFunction') || strcmp(ME.identifier, 'MATLAB:class:UndefinedClass'))
                fprintf('✓ 创建 %s 智能体: %s (%s)\n', agent_type, name, algorithm);
            end
        end
        
        function agents = createMultipleAgents(algorithms, agent_type, config, state_dim, action_dim)
            % 批量创建智能体
            
            n_agents = length(algorithms);
            agents = cell(1, n_agents);
            
            for i = 1:n_agents
                name = sprintf('%s_%d_%s', agent_type, i, algorithms{i});
                agents{i} = AgentFactory.createAgent(algorithms{i}, name, agent_type, ...
                                                   config, state_dim, action_dim);
            end
        end
        
        function validateAgent(agent)
            % 验证智能体配置是否正确
            
            % 检查必要属性
            required_properties = {'name', 'agent_type', 'state_dim', 'action_dim', ...
                                 'learning_rate', 'discount_factor'};
            
            for i = 1:length(required_properties)
                prop = required_properties{i};
                if ~isprop(agent, prop) || isempty(agent.(prop))
                    error('AgentFactory:InvalidAgent', ...
                          '智能体缺少必要属性: %s', prop);
                end
            end
            
            % 检查探索策略相关属性
            if isprop(agent, 'exploration_strategy')
                switch agent.exploration_strategy
                    case 'epsilon-greedy'
                        assert(isprop(agent, 'epsilon'), '缺少epsilon属性');
                        assert(agent.epsilon >= 0 && agent.epsilon <= 1, ...
                               'epsilon必须在[0,1]范围内');
                        
                    case 'softmax'
                        assert(isprop(agent, 'temperature'), '缺少temperature属性');
                        assert(agent.temperature > 0, 'temperature必须大于0');
                end
            end
            
            % 检查数值范围
            assert(agent.learning_rate > 0 && agent.learning_rate <= 1, ...
                   '学习率必须在(0,1]范围内');
            assert(agent.discount_factor >= 0 && agent.discount_factor <= 1, ...
                   '折扣因子必须在[0,1]范围内');
        end
        
        function config = ensureConfigCompleteness(config)
            % 确保配置包含所有必要参数
            
            % 如果有新格式的rl_defaults，使用它
            if isfield(config, 'rl_defaults')
                rl_defaults = config.rl_defaults;
                
                % 基本参数
                if ~isfield(config, 'learning_rate')
                    config.learning_rate = rl_defaults.learning_rate;
                end
                if ~isfield(config, 'discount_factor')
                    config.discount_factor = rl_defaults.discount_factor;
                end
                
                % 探索策略
                if ~isfield(config, 'exploration_strategy')
                    config.exploration_strategy = rl_defaults.exploration_strategy;
                end
                
                % 根据探索策略设置相应参数
                if strcmp(config.exploration_strategy, 'epsilon-greedy')
                    eps_params = rl_defaults.epsilon_greedy;
                    if ~isfield(config, 'epsilon')
                        config.epsilon = eps_params.epsilon;
                    end
                    if ~isfield(config, 'epsilon_decay')
                        config.epsilon_decay = eps_params.epsilon_decay;
                    end
                    if ~isfield(config, 'epsilon_min')
                        config.epsilon_min = eps_params.epsilon_min;
                    end
                elseif strcmp(config.exploration_strategy, 'softmax')
                    temp_params = rl_defaults.softmax_exploration;
                    if ~isfield(config, 'temperature')
                        config.temperature = temp_params.temperature;
                    end
                    if ~isfield(config, 'temperature_decay')
                        config.temperature_decay = temp_params.temperature_decay;
                    end
                    if ~isfield(config, 'temperature_min')
                        config.temperature_min = temp_params.temperature_min;
                    end
                end
            else
                % 使用传统默认值
                config = AgentFactory.applyLegacyDefaults(config);
            end
            
            % 学习率调度参数
            if ~isfield(config, 'learning_rate_decay')
                config.learning_rate_decay = 0.9995;
            end
            if ~isfield(config, 'learning_rate_min')
                config.learning_rate_min = 0.001;
            end
        end
        
        function config = applyLegacyDefaults(config)
            % 应用传统默认值（向后兼容）
            
            % 基本参数
            if ~isfield(config, 'learning_rate')
                config.learning_rate = 0.1;
            end
            if ~isfield(config, 'discount_factor')
                config.discount_factor = 0.95;
            end
            
            % 默认使用epsilon-greedy
            if ~isfield(config, 'exploration_strategy')
                config.exploration_strategy = 'epsilon-greedy';
            end
            
            % Epsilon-Greedy参数
            if ~isfield(config, 'epsilon')
                config.epsilon = 0.3;
            end
            if ~isfield(config, 'epsilon_decay')
                config.epsilon_decay = 0.995;
            end
            if ~isfield(config, 'epsilon_min')
                config.epsilon_min = 0.01;
            end
            
            % Softmax参数
            if ~isfield(config, 'temperature')
                config.temperature = 1.0;
            end
            if ~isfield(config, 'temperature_decay')
                config.temperature_decay = 0.995;
            end
            if ~isfield(config, 'temperature_min')
                config.temperature_min = 0.1;
            end
        end
        
        function configureAgentSpecifics(agent, algorithm, config)
            % 配置智能体特定参数
            
            switch upper(algorithm)
                case 'QLEARNING'
                    % Q-Learning特定配置
                    if isprop(agent, 'use_double_q')
                        agent.use_double_q = false;
                    end
                    
                case 'SARSA'
                    % SARSA特定配置 - 更保守的探索
                    if strcmp(agent.exploration_strategy, 'epsilon-greedy')
                        agent.epsilon = agent.epsilon * 0.8;  % 降低探索率
                    elseif strcmp(agent.exploration_strategy, 'softmax')
                        agent.temperature = agent.temperature * 1.2;  % 提高温度
                    end
                    
                case 'DOUBLEQLEARNING'
                    % Double Q-Learning特定配置
                    if isprop(agent, 'use_double_q')
                        agent.use_double_q = true;
                    end
                    if isprop(agent, 'q1_weight')
                        agent.q1_weight = 0.5;
                        agent.q2_weight = 0.5;
                    end
                case 'DQN' % DQN 特定配置
                    % 可以在此处添加 DQN 特有的配置，例如网络结构、经验回放缓冲区大小等
                    % 假设 DQNAgent 类有这些属性
                    if isprop(agent, 'memory_size') && isfield(config, 'learning') && isfield(config.learning, 'memory_size')
                        agent.memory_size = config.learning.memory_size;
                    end
                    if isprop(agent, 'batch_size') && isfield(config, 'learning') && isfield(config.learning, 'batch_size')
                        agent.batch_size = config.learning.batch_size;
                    end
            end
        end
        
        function attacker_config = configureAttackerParams(config)
            % 为攻击者配置特殊参数
            attacker_config = config;
            
            % 攻击者通常需要更高的探索率
            if strcmp(attacker_config.exploration_strategy, 'epsilon-greedy')
                if isfield(attacker_config, 'epsilon')
                    attacker_config.epsilon = min(0.8, attacker_config.epsilon * 1.5);
                end
            elseif strcmp(attacker_config.exploration_strategy, 'softmax')
                if isfield(attacker_config, 'temperature')
                    attacker_config.temperature = attacker_config.temperature * 1.2;
                end
            end
            
            % 攻击者可能需要不同的学习率
            if isfield(attacker_config, 'learning_rate')
                attacker_config.learning_rate = attacker_config.learning_rate * 1.2;
            end
        end
        
        function configureAttackerSpecifics(agent, config)
            % 配置攻击者特定属性
            
            if isprop(agent, 'exploration_bonus')
                agent.exploration_bonus = 0.1;  % 攻击者获得探索奖励
            end
            
            if isprop(agent, 'risk_tolerance')
                agent.risk_tolerance = 0.8;  % 攻击者风险容忍度更高
            end
            
            % 攻击者可能使用不同的动作选择策略
            if isprop(agent, 'action_selection_mode')
                agent.action_selection_mode = 'aggressive';
            end
        end
        
        function displayAgentInfo(agent)
            % 显示智能体信息
            
            fprintf('\n智能体信息:\n');
            fprintf('  名称: %s\n', agent.name);
            fprintf('  类型: %s\n', agent.agent_type);
            fprintf('  算法: %s\n', class(agent));
            fprintf('  状态维度: %d\n', agent.state_dim);
            fprintf('  动作维度: %d\n', agent.action_dim);
            fprintf('  学习率: %.4f\n', agent.learning_rate);
            fprintf('  折扣因子: %.4f\n', agent.discount_factor);
            
            if isprop(agent, 'exploration_strategy')
                fprintf('  探索策略: %s\n', agent.exploration_strategy);
                
                switch agent.exploration_strategy
                    case 'epsilon-greedy'
                        fprintf('    Epsilon: %.4f\n', agent.epsilon);
                        fprintf('    Epsilon衰减: %.4f\n', agent.epsilon_decay);
                        fprintf('    最小Epsilon: %.4f\n', agent.epsilon_min);
                        
                    case 'softmax'
                        fprintf('    温度: %.4f\n', agent.temperature);
                        fprintf('    温度衰减: %.4f\n', agent.temperature_decay);
                        fprintf('    最小温度: %.4f\n', agent.temperature_min);
                end
            end
            
            fprintf('\n');
        end
    end
end
