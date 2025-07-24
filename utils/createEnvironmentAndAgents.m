function [env, attacker_agent, defender_agents] = createEnvironmentAndAgents(config)
    %% createEnvironmentAndAgents - 创建仿真环境和智能体
    % 输入: config - 配置结构体
    % 输出: env - 环境对象
    %       attacker_agent - 攻击者智能体
    %       defender_agents - 防御者智能体数组
    
    fprintf('🏗️ 创建仿真环境和智能体...\n');
    
    try
        %% 1. 创建TCS环境
        env = TCSEnvironment(config);
        fprintf('✓ TCS 环境创建成功。\n');
        Logger.info('TCS 环境创建成功。');
        
        %% 2. 获取状态和动作空间维度
        if isfield(config, 'system')
            state_dim = config.system.state_space_size;
            action_dim = config.system.action_space_size;
        else
            state_dim = 77;  % 默认状态空间大小
            action_dim = 20; % 默认动作空间大小
            fprintf('⚠️ 使用默认的状态空间(%d)和动作空间(%d)大小\n', state_dim, action_dim);
        end
        
        %% 3. 创建攻击者智能体
        attacker_algorithm = 'QLearning'; % 默认使用Q-Learning
        if isfield(config, 'algorithms') && isfield(config.algorithms, 'attacker')
            attacker_algorithm = config.algorithms.attacker;
        end
        
        try
            attacker_agent = AgentFactory.createAgent('Attacker', 'attacker', ...
                                                    attacker_algorithm, config, ...
                                                    state_dim, action_dim);
            fprintf('✓ 攻击者智能体 "%s" 创建成功。\n', attacker_algorithm);
            Logger.info(sprintf('攻击者智能体 "%s" 创建成功。', attacker_algorithm));
        catch ME
            fprintf('❌ 创建攻击者智能体失败: %s\n', ME.message);
            Logger.error(sprintf('创建攻击者智能体 "%s" 失败: %s', attacker_algorithm, ME.message));
            
            % 创建备用攻击者
            attacker_agent = QLearningAgent('Attacker_Fallback', 'attacker', config, state_dim, action_dim);
            fprintf('✓ 备用 QLearning 攻击者智能体创建成功。\n');
        end
        
        %% 4. 创建防御者智能体
        defender_algorithms = {'QLearning', 'SARSA'}; % 默认算法
        if isfield(config, 'algorithms') && isfield(config.algorithms, 'defender')
            defender_algorithms = config.algorithms.defender;
        end
        
        defender_agents = {};
        for i = 1:length(defender_algorithms)
            algorithm = defender_algorithms{i};
            agent_name = sprintf('Defender_%s', algorithm);
            
            try
                defender_agent = AgentFactory.createAgent(agent_name, 'defender', ...
                                                        algorithm, config, ...
                                                        state_dim, action_dim);
                defender_agents{end+1} = defender_agent;
                fprintf('✓ 防御者智能体 "%s" 创建成功。\n', algorithm);
                Logger.info(sprintf('防御者智能体 "%s" 创建成功。', algorithm));
                
            catch ME
                fprintf('❌ 创建防御者智能体 "%s" 失败: %s\n', algorithm, ME.message);
                Logger.error(sprintf('创建防御者智能体 "%s" 失败: %s', algorithm, ME.message));
                
                % 创建备用防御者
                try
                    fallback_name = sprintf('Defender_%s_Fallback', algorithm);
                    fallback_agent = QLearningAgent(fallback_name, 'defender', config, state_dim, action_dim);
                    defender_agents{end+1} = fallback_agent;
                    fprintf('✓ 备用 QLearning 防御者智能体 "%s" 创建成功。\n', fallback_name);
                    Logger.info(sprintf('备用 QLearning 防御者智能体 "%s" 创建成功。', fallback_name));
                catch ME2
                    fprintf('❌ 创建备用防御者也失败: %s\n', ME2.message);
                end
            end
        end
        
        %% 5. 验证创建结果
        if isempty(defender_agents)
            error('所有防御者智能体创建都失败了');
        end
        
        fprintf('✅ 环境和智能体创建完成: 1个攻击者, %d个防御者\n', length(defender_agents));
        
    catch ME
        fprintf('❌ 环境或智能体创建过程中发生致命错误: %s\n', ME.message);
        Logger.error(sprintf('环境或智能体创建过程中发生致命错误: %s', ME.message));
        
        % 返回空值，让调用者处理
        env = [];
        attacker_agent = [];
        defender_agents = {};
        rethrow(ME);
    end
end