%% AgentFactory.m - 智能体工厂类
% =========================================================================
% 描述: 统一创建和管理不同类型的强化学习智能体
% =========================================================================

classdef AgentFactory
    
    methods (Static)
        function agents = createDefenderAgents(config, environment)
            % 创建防御者智能体数组
            % 输入: config - 配置结构体, environment - 环境对象
            % 输出: agents - 智能体对象数组
            
            fprintf('🛡️ 创建防御者智能体...\n');
            
            % 验证输入
            if ~isstruct(config)
                error('AgentFactory:InvalidInput', '配置参数必须是结构体');
            end
            
            % 获取算法列表
            if isfield(config, 'algorithms')
                algorithms = config.algorithms;
            elseif isfield(config, 'defender_types')
                algorithms = config.defender_types;
            else
                algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
                fprintf('使用默认算法: %s\n', strjoin(algorithms, ', '));
            end
            
            % 获取状态和动作维度
            if ismethod(environment, 'getStateDimension') && ismethod(environment, 'getActionDimension')
                state_dim = environment.getStateDimension();
                action_dim = environment.getActionDimension();
            else
                % 默认维度
                state_dim = 1024;  % 2^10 for binary state encoding
                action_dim = 10;   % 10 possible defense actions
                fprintf('[DEBUG] 使用默认维度: state_dim=%d, action_dim=%d\n', state_dim, action_dim);
            end
            
            % 创建智能体数组
            agents = cell(1, length(algorithms));
            
            for i = 1:length(algorithms)
                algorithm = algorithms{i};
                agent_name = sprintf('Defender_%d_%s', i, algorithm);
                
                fprintf('  创建 %s 智能体...\n', algorithm);
                
                % 确保配置完整性
                agent_config = AgentFactory.ensureCompleteConfig(config);
                
                % 根据算法类型创建智能体
                switch upper(algorithm)
                    case 'QLEARNING'
                        agents{i} = QLearningAgent(agent_name, 'defender', agent_config, state_dim, action_dim);
                    case 'SARSA'
                        agents{i} = SARSAAgent(agent_name, 'defender', agent_config, state_dim, action_dim);
                    case 'DOUBLEQLEARNING'
                        agents{i} = DoubleQLearningAgent(agent_name, 'defender', agent_config, state_dim, action_dim);
                    otherwise
                        warning('AgentFactory:UnknownAlgorithm', '未知算法 %s，创建默认Q-Learning智能体', algorithm);
                        agents{i} = QLearningAgent(agent_name, 'defender', agent_config, state_dim, action_dim);
                end
                
                % 设置智能体特定参数
                AgentFactory.configureAgentSpecifics(agents{i}, algorithm, agent_config);
                
                fprintf('    ✓ %s 创建完成\n', agent_name);
            end
            
            fprintf('✓ 防御者智能体创建完成，共 %d 个\n', length(agents));
        end
        
        function agent = createAttackerAgent(config, environment)
            % 创建攻击者智能体
            % 输入: config - 配置结构体, environment - 环境对象
            % 输出: agent - 攻击者智能体对象
            
            fprintf('⚔️ 创建攻击者智能体...\n');
            
            % 获取攻击者算法
            if isfield(config, 'attacker_algorithm')
                algorithm = config.attacker_algorithm;
            else
                algorithm = 'QLearning';
                fprintf('使用默认攻击者算法: %s\n', algorithm);
            end
            
            % 获取状态和动作维度
            if ismethod(environment, 'getStateDimension') && ismethod(environment, 'getActionDimension')
                state_dim = environment.getStateDimension();
                action_dim = environment.getActionDimension();
            else
                % 攻击者可能有不同的动作空间
                state_dim = 1024;
                action_dim = 8;   % 8 possible attack actions
                fprintf('[DEBUG] 使用默认攻击者维度: state_dim=%d, action_dim=%d\n', state_dim, action_dim);
            end
            
            agent_name = sprintf('Attacker_%s', algorithm);
            
            % 确保配置完整性
            agent_config = AgentFactory.ensureCompleteConfig(config);
            
            % 攻击者可能有不同的参数设置
            agent_config = AgentFactory.configureAttackerParams(agent_config);
            
            % 创建攻击者智能体
            switch upper(algorithm)
                case 'QLEARNING'
                    agent = QLearningAgent(agent_name, 'attacker', agent_config, state_dim, action_dim);
                case 'SARSA'
                    agent = SARSAAgent(agent_name, 'attacker', agent_config, state_dim, action_dim);
                case 'DOUBLEQLEARNING'
                    agent = DoubleQLearningAgent(agent_name, 'attacker', agent_config, state_dim, action_dim);
                otherwise
                    warning('AgentFactory:UnknownAlgorithm', '未知攻击者算法 %s，创建默认Q-Learning智能体', algorithm);
                    agent = QLearningAgent(agent_name, 'attacker', agent_config, state_dim, action_dim);
            end
            
            % 设置攻击者特定参数
            AgentFactory.configureAttackerSpecifics(agent, agent_config);
            
            fprintf('✓ 攻击者智能体创建完成: %s\n', agent_name);
        end
        
        function complete_config = ensureCompleteConfig(config)
            % 确保配置结构体包含所有必需的字段
            complete_config = config;
            
            % 学习参数
            if ~isfield(complete_config, 'learning_rate')
                complete_config.learning_rate = 0.15;
            end
            
            if ~isfield(complete_config, 'discount_factor')
                complete_config.discount_factor = 0.95;
            end
            
            if ~isfield(complete_config, 'epsilon')
                complete_config.epsilon = 0.4;
            end
            
            if ~isfield(complete_config, 'epsilon_decay')
                complete_config.epsilon_decay = 0.999;
            end
            
            if ~isfield(complete_config, 'epsilon_min')
                complete_config.epsilon_min = 0.05;
            end
            
            if ~isfield(complete_config, 'temperature')
                complete_config.temperature = 1.0;
            end
            
            if ~isfield(complete_config, 'temperature_decay')
                complete_config.temperature_decay = 0.995;
            end
            
            if ~isfield(complete_config, 'temperature_min')
                complete_config.temperature_min = 0.1;
            end
            
            % 调试信息
            fprintf('[DEBUG] 配置完整性检查完成，learning_rate = %.3f\n', complete_config.learning_rate);
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
                    if isprop(agent, 'epsilon')
                        agent.epsilon = agent.epsilon * 0.8;  % 降低探索率
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
            end
        end
        
        function attacker_config = configureAttackerParams(config)
            % 为攻击者配置特殊参数
            attacker_config = config;
            
            % 攻击者通常需要更高的探索率
            if isfield(attacker_config, 'epsilon')
                attacker_config.epsilon = min(0.8, attacker_config.epsilon * 1.5);
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
        end
    end
end