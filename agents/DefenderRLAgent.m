classdef DefenderRLAgent < RLAgent
    % 专门为防御者设计的RL智能体
    
    properties
        n_stations
        total_resources
        allocation_strategy = 'proportional' % 'proportional', 'threshold', 'mixed'
    end
    
    methods
        function obj = DefenderRLAgent(name, agent_type, config, state_dim, action_dim)
            obj = obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.n_stations = config.system.n_stations;
            obj.total_resources = config.system.total_resources;
        end
        
        function deployment = selectAction(obj, state)
            % 选择离散动作，然后转换为资源分配
            
            % 1. 获取离散动作（使用父类方法）
            discrete_action = selectAction@RLAgent(obj, state);
            
            % 2. 将离散动作转换为资源分配策略
            deployment = obj.convertToDeployment(discrete_action, state);
        end
        
        function deployment = convertToDeployment(obj, action, state)
            % 将离散动作转换为实际的资源部署
            
            % 提取状态中的威胁信息
            threat_levels = state(1:obj.n_stations); % 攻击者平均策略
            recent_attacks = zeros(1, obj.n_stations);
            if length(state) >= 2*obj.n_stations
                recent_attacks = state(obj.n_stations+1:2*obj.n_stations);
            end
            
            % 组合威胁评估
            combined_threat = 0.7 * threat_levels + 0.3 * recent_attacks;
            
            % 基于动作选择分配策略
            switch obj.allocation_strategy
                case 'proportional'
                    % 按威胁比例分配
                    deployment = obj.proportionalAllocation(combined_threat);
                    
                case 'threshold'
                    % 阈值分配
                    deployment = obj.thresholdAllocation(combined_threat, action);
                    
                case 'mixed'
                    % 混合策略
                    deployment = obj.mixedAllocation(combined_threat, action);
                    
                otherwise
                    deployment = ones(1, obj.n_stations) * obj.total_resources / obj.n_stations;
            end
            
            % 确保资源约束
            deployment = obj.enforceResourceConstraints(deployment);
        end
        
        function deployment = proportionalAllocation(obj, threat_levels)
            % 按威胁比例分配资源
            if sum(threat_levels) > 0
                deployment = threat_levels / sum(threat_levels) * obj.total_resources;
            else
                deployment = ones(1, obj.n_stations) * obj.total_resources / obj.n_stations;
            end
        end
        
        function deployment = thresholdAllocation(obj, threat_levels, focus_station)
            % 阈值分配：主要资源给重点站点
            deployment = ones(1, obj.n_stations) * 2; % 基础防御
            
            % 60%资源给重点站点
            main_resources = obj.total_resources * 0.6;
            deployment(focus_station) = main_resources;
            
            % 剩余资源按威胁分配
            remaining = obj.total_resources - sum(deployment);
            if remaining > 0 && sum(threat_levels) > 0
                extra = threat_levels / sum(threat_levels) * remaining;
                deployment = deployment + extra;
            end
        end
        
        function deployment = mixedAllocation(obj, threat_levels, action)
            % 混合策略：结合集中和分散
            
            % 解析动作：前5位表示集中防御，后5位表示分散防御
            if action <= 5
                % 集中防御模式
                focus = action;
                deployment = obj.thresholdAllocation(threat_levels, focus);
            else
                % 分散防御模式
                deployment = obj.proportionalAllocation(threat_levels);
                
                % 略微加强某些站点
                boost_station = action - 5;
                if boost_station <= obj.n_stations
                    deployment(boost_station) = deployment(boost_station) * 1.2;
                end
            end
        end
        
        function deployment = enforceResourceConstraints(obj, deployment)
            % 确保资源约束
            deployment = max(0, deployment); % 非负
            
            % 总量约束
            if sum(deployment) > obj.total_resources
                deployment = deployment * (obj.total_resources / sum(deployment));
            end
        end
    end
endcomputeRewards