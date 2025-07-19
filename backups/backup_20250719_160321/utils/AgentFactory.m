%% AgentFactory.m - 智能体工厂类
classdef AgentFactory
    methods (Static)
        function agents = createDefenderAgents(config, env)
            agents = {};
            for i = 1:length(config.defender_types)
                switch config.defender_types{i}
                    case 'QLearning'
                        agents{i} = QLearningAgent(sprintf('Q-Learning-%d', i), ...
                                                  'defender', config.agents.defenders{i}, ...
                                                  env.state_dim, env.action_dim);
                    case 'SARSA'
                        agents{i} = SARSAAgent(sprintf('SARSA-%d', i), ...
                                             'defender', config.agents.defenders{i}, ...
                                             env.state_dim, env.action_dim);
                    case 'DoubleQLearning'
                        agents{i} = DoubleQLearningAgent(sprintf('DoubleQ-%d', i), ...
                                                       'defender', config.agents.defenders{i}, ...
                                                       env.state_dim, env.action_dim);
                    otherwise
                        error('未知算法类型: %s', config.defender_types{i});
                end
            end
        end
        
        function agent = createAttackerAgent(config, env)
            agent = QLearningAgent('Attacker', 'attacker', config.agents.attacker, ...
                                 env.state_dim, env.action_dim);
        end
    end
end
