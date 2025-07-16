%% RLAgent.m - 强化学习智能体基类
% =========================================================================
% 描述: 所有RL智能体的抽象基类
% =========================================================================

classdef (Abstract) RLAgent < handle
    
    properties (Access = public)
        name                % 智能体名称
        agent_type          % 智能体类型
        state_dim           % 状态空间维度
        action_dim          % 动作空间维度
        config              % 配置参数
        
        % 学习参数
        learning_rate       % 学习率
        discount_factor     % 折扣因子
        epsilon             % 探索率
        epsilon_min         % 最小探索率
        epsilon_decay       % 探索率衰减
        
        % 附加参数
        temperature         % 温度参数（用于softmax）
        temperature_decay   % 温度衰减
        temperature_min     % 最小温度
        learning_rate_min   % 最小学习率
        learning_rate_decay % 学习率衰减
    end
    
    methods (Abstract)
        % 抽象方法 - 子类必须实现
        action = selectAction(obj, state)
        update(obj, state, action, reward, next_state, next_action)
    end
    
    methods
        function obj = RLAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            
            obj.name = name;
            obj.agent_type = agent_type;
            obj.config = config;
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            
            % 从配置中提取参数
            if isfield(config, 'learning_rate')
                obj.learning_rate = config.learning_rate;
            else
                obj.learning_rate = 0.1;
            end
            
            if isfield(config, 'discount_factor')
                obj.discount_factor = config.discount_factor;
            else
                obj.discount_factor = 0.95;
            end
            
            if isfield(config, 'epsilon')
                obj.epsilon = config.epsilon;
            else
                obj.epsilon = 0.3;
            end
            
            if isfield(config, 'epsilon_min')
                obj.epsilon_min = config.epsilon_min;
            else
                obj.epsilon_min = 0.01;
            end
            
            if isfield(config, 'epsilon_decay')
                obj.epsilon_decay = config.epsilon_decay;
            else
                obj.epsilon_decay = 0.995;
            end
            
            % 初始化附加参数
            obj.temperature = 1.0;
            obj.temperature_decay = 0.995;
            obj.temperature_min = 0.1;
            obj.learning_rate_min = 0.001;
            obj.learning_rate_decay = 0.9995;
        end
        
        function reset(obj)
            % 重置智能体（可被子类覆盖）
            % 默认实现：不做任何操作
        end
        
        function info = getInfo(obj)
            % 获取智能体信息
            info = struct();
            info.name = obj.name;
            info.type = obj.agent_type;
            info.learning_rate = obj.learning_rate;
            info.epsilon = obj.epsilon;
            info.state_dim = obj.state_dim;
            info.action_dim = obj.action_dim;
        end
        
        function updateParameters(obj)
            % 更新学习参数（衰减等）
            
            % 更新探索率
            obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
            
            % 更新学习率
            obj.learning_rate = max(obj.learning_rate_min, ...
                                   obj.learning_rate * obj.learning_rate_decay);
            
            % 更新温度
            obj.temperature = max(obj.temperature_min, ...
                                 obj.temperature * obj.temperature_decay);
        end
    end
end