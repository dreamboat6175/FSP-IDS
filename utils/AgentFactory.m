%% AgentFactory.m - 智能体工厂类
classdef AgentFactory
    methods (Static)
        function agents = createDefenderAgents(config, env)
            agents = {};
            for i = 1:length(config.algorithms)
                switch config.algorithms{i}
                    case 'Q-Learning'
                        agents{i} = QLearningAgent(sprintf('Q-Learning-%d', i), ...
                                                  'defender', config, ...
                                                  env.state_dim, env.action_dim_defender);
                    case 'SARSA'
                        agents{i} = SARSAAgent(sprintf('SARSA-%d', i), ...
                                             'defender', config, ...
                                             env.state_dim, env.action_dim_defender);
                    case 'Double Q-Learning'
                        agents{i} = DoubleQLearningAgent(sprintf('DoubleQ-%d', i), ...
                                                       'defender', config, ...
                                                       env.state_dim, env.action_dim_defender);
                    otherwise
                        error('未知算法类型: %s', config.algorithms{i});
                end
            end
        end
        
        function agent = createAttackerAgent(config, env)
            agent = QLearningAgent('Attacker', 'attacker', config, ...
                                 env.state_dim, env.action_dim_attacker);
        end
    end
end
