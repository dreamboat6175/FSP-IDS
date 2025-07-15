%% RLAgent.m - 强化学习智能体基类
% =========================================================================
% 描述: 强化学习智能体的抽象基类，定义统一接口
% 优化要点：
% 1. 统一的接口设计
% 2. 参数验证和默认值
% 3. 扩展性设计
% =========================================================================

classdef (Abstract) RLAgent < handle
    
    properties (Access = protected)
        name                % 智能体名称
        agent_type          % 智能体类型
        state_dim           % 状态维度
        action_dim          % 动作维度
        learning_rate       % 学习率
        discount_factor     % 折扣因子
        epsilon             % 探索率
        update_count        % 更新次数
        total_reward        % 累计奖励
        episode_rewards     % 每轮奖励历史
        action_history      % 动作历史
    end
    
    methods (Access = public)
        function obj = RLAgent(name, agent_type, config, state_dim, action_dim)
            % 基类构造函数
            
            obj.name = name;
            obj.agent_type = agent_type;
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            
            % 从配置设置参数
            obj.setConfigParameters(config);
            
            % 初始化统计
            obj.initializeStatistics();
        end
        
        function stats = getStatistics(obj)
            % 获取基础统计信息
            
            stats = struct();
            stats.name = obj.name;
            stats.agent_type = obj.agent_type;
            stats.update_count = obj.update_count;
            stats.total_reward = obj.total_reward;
            stats.avg_episode_reward = obj.getAverageEpisodeReward();
            stats.current_epsilon = obj.epsilon;
            stats.current_learning_rate = obj.learning_rate;
        end
    end
    
    methods (Abstract, Access = public)
        % 抽象方法 - 子类必须实现
        action = selectAction(obj, state);
        update(obj, state, action, reward, next_state, next_action);
    end
    
    methods (Access = protected)
        function setConfigParameters(obj, config)
            % 从配置设置参数
            
            obj.learning_rate = obj.getConfigValue(config, 'learning_rate', 0.1);
            obj.discount_factor = obj.getConfigValue(config, 'discount_factor', 0.95);
            obj.epsilon = obj.getConfigValue(config, 'epsilon', 0.1);
        end
        
        function value = getConfigValue(obj, config, field_name, default_value)
            % 安全地从配置获取值
            
            if isfield(config, field_name)
                value = config.(field_name);
            else
                value = default_value;
            end
        end
        
        function initializeStatistics(obj)
            % 初始化统计信息
            
            obj.update_count = 0;
            obj.total_reward = 0;
            obj.episode_rewards = [];
            obj.action_history = [];
        end
        
        function avg_reward = getAverageEpisodeReward(obj)
            % 计算平均轮次奖励
            
            if isempty(obj.episode_rewards)
                avg_reward = 0;
            else
                avg_reward = mean(obj.episode_rewards);
            end
        end
    end
end