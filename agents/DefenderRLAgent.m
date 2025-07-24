classdef DefenderRLAgent < RLAgent
    % 专门为防御者设计的RL智能体 - 修复版
    
    properties
        n_stations
        total_resources
        allocation_strategy = 'progressive' % 新增：渐进式策略
        learning_phase = 'exploration'      % 学习阶段：exploration -> exploitation
        phase_transition_step = 100         % 阶段转换步数
    end
    
    methods
        function obj = DefenderRLAgent(name, agent_type, config, state_dim, action_dim)
            obj = obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.n_stations = config.system.n_stations;
            obj.total_resources = config.system.total_resources;
            
            % 初始化学习参数（确保初期高探索率）
            obj.epsilon = 0.8;  % 高初始探索率
            obj.epsilon_min = 0.1;  % 合理的最小探索率
            obj.epsilon_decay = 0.998;  % 缓慢衰减
        end
        
        function deployment = selectAction(obj, state)
            % 选择离散动作，然后转换为资源分配
            
            % 1. 获取离散动作（使用父类方法）
            discrete_action = selectAction@RLAgent(obj, state);
            
            % 2. 将离散动作转换为资源分配策略
            deployment = obj.convertToDeployment(discrete_action, state);
        end
        
        function deployment = convertToDeployment(obj, action, state)
            % 将离散动作转换为实际的资源部署 - 修复版
            
            % 提取状态中的威胁信息
            threat_levels = state(1:obj.n_stations); % 攻击者平均策略
            recent_attacks = zeros(1, obj.n_stations);
            if length(state) >= 2*obj.n_stations
                recent_attacks = state(obj.n_stations+1:2*obj.n_stations);
            end
            
            % 更新学习阶段
            if obj.update_count > obj.phase_transition_step
                obj.learning_phase = 'exploitation';
                obj.allocation_strategy = 'adaptive';
            end
            
            % 基于学习阶段选择策略
            switch obj.learning_phase
                case 'exploration'
                    % 早期：均匀分配为主，逐渐引入变化
                    deployment = obj.explorationPhaseAllocation(threat_levels, action);
                    
                case 'exploitation'
                    % 后期：基于学习的自适应分配
                    deployment = obj.exploitationPhaseAllocation(threat_levels, recent_attacks, action);
            end
            
            % 确保资源约束
            deployment = obj.enforceResourceConstraints(deployment);
        end
        
        function deployment = explorationPhaseAllocation(obj, threat_levels, action)
            % 探索阶段的资源分配（初期保持均匀）
            
            % 基础均匀分配
            base_allocation = ones(1, obj.n_stations) * obj.total_resources / obj.n_stations;
            
            % 逐渐引入基于威胁的调整
            progress = min(1.0, obj.update_count / obj.phase_transition_step);
            
            if progress < 0.3
                % 前30%时间：完全均匀分配
                deployment = base_allocation;
            else
                % 逐渐引入威胁响应
                threat_response = zeros(1, obj.n_stations);
                if sum(threat_levels) > 0
                    threat_response = threat_levels / sum(threat_levels) * obj.total_resources;
                end
                
                % 混合均匀分配和威胁响应
                mix_factor = (progress - 0.3) / 0.7;  % 从0到1的过渡
                deployment = (1 - mix_factor) * base_allocation + mix_factor * threat_response;
            end
        end
        
        function deployment = exploitationPhaseAllocation(obj, threat_levels, recent_attacks, action)
            % 利用阶段的资源分配（基于学习）
            
            % 组合威胁评估
            combined_threat = 0.7 * threat_levels + 0.3 * recent_attacks;
            
            % 基于动作选择分配策略
            if action <= obj.n_stations
                % 集中防御特定站点
                deployment = obj.focusedDefense(combined_threat, action);
            else
                % 分布式防御
                deployment = obj.distributedDefense(combined_threat);
            end
        end
        
        function deployment = focusedDefense(obj, threat_assessment, focus_station)
            % 集中防御策略（但不过度集中）
            deployment = ones(1, obj.n_stations) * obj.total_resources * 0.05; % 基础防御
            
            % 主要资源给重点站点（但不超过40%）
            main_resources = obj.total_resources * 0.4;
            deployment(focus_station) = main_resources;
            
            % 剩余资源基于威胁分配
            remaining = obj.total_resources - sum(deployment);
            if remaining > 0 && sum(threat_assessment) > 0
                threat_normalized = threat_assessment / sum(threat_assessment);
                deployment = deployment + threat_normalized * remaining;
            end
        end
        
        function deployment = distributedDefense(obj, threat_assessment)
            % 分布式防御策略
            
            % 基础均匀分配（30%）
            base_allocation = ones(1, obj.n_stations) * obj.total_resources * 0.3 / obj.n_stations;
            
            % 基于威胁的分配（70%）
            threat_allocation = zeros(1, obj.n_stations);
            if sum(threat_assessment) > 0
                threat_normalized = threat_assessment / sum(threat_assessment);
                threat_allocation = threat_normalized * obj.total_resources * 0.7;
            else
                threat_allocation = ones(1, obj.n_stations) * obj.total_resources * 0.7 / obj.n_stations;
            end
            
            deployment = base_allocation + threat_allocation;
        end
        
        function deployment = enforceResourceConstraints(obj, deployment)
            % 确保资源约束
            deployment = max(0, deployment); % 非负
            
            % 确保每个站点至少有最小资源
            min_per_station = obj.total_resources * 0.01;
            deployment = max(deployment, min_per_station);
            
            % 总量约束
            if sum(deployment) > obj.total_resources
                deployment = deployment * (obj.total_resources / sum(deployment));
            end
        end
        
        function updateLearningPhase(obj)
            % 更新学习阶段
            if obj.update_count > obj.phase_transition_step * 2
                obj.allocation_strategy = 'advanced';
            elseif obj.update_count > obj.phase_transition_step
                obj.allocation_strategy = 'adaptive';
            end
        end
    end
end