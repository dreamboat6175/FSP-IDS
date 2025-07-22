classdef TCSEnvironment < handle
    %% TCSEnvironment - 交通控制系统环境（完整优化版）
    % ================================================================
    % 版本：v4.0 - 增强版数据记录与可视化支持
    % 新增功能：
    % 1. RADI、Nash均衡收敛度、攻击覆盖率数据记录
    % 2. 策略变化跟踪
    % 3. 防御有效性评估
    % 4. 数据完整性验证
    % ================================================================
    
    properties (Access = public)
        %% === 基础系统参数 ===
        n_stations              % 站点数量
        n_components            % 各站点组件数量向量
        total_components        % 总组件数
        total_resources         % 总资源数
        n_resource_types        % 资源类型数
        n_attack_types          % 攻击类型数
        station_values          % 站点价值向量
        
        %% === 环境状态相关 ===
        time_step               % 当前时间步
        current_state           % 当前状态向量
        state_dim               % 状态维度
        action_dim              % 动作维度
        action_dim_defender     % 防御者动作维度
        action_dim_attacker     % 攻击者动作维度
        
        %% === FSP相关参数 ===
        alpha_ewma              % FSP指数加权移动平均参数
        attacker_avg_strategy   % 攻击者平均策略
        
        %% === 基线Q学习参数（内置攻击者） ===
        attacker_lr             % 攻击者学习率
        attacker_gamma          % 攻击者折扣因子
        attacker_epsilon        % 攻击者探索率
        attacker_epsilon_decay  % 探索率衰减
        attacker_epsilon_min    % 最小探索率
        max_defense_states      % 最大防御状态数
        attacker_Q_table        % 攻击者Q表
        
        %% === RADI计算参数 ===
        radi_score              % 当前RADI分数
        radi_config             % RADI配置参数
        
        %% === 检测系统参数 ===
        detection_enabled       % 是否启用检测系统
        base_detection_rate     % 基础检测率
        detection_sensitivity   % 检测敏感度
        false_positive_rate     % 误报率
        
        %% === 历史记录（原有） ===
        attack_success_history     % 攻击成功历史 [0,1]
        attack_target_history      % 攻击目标历史 [one-hot]
        defense_deployment_history % 防御部署历史
        damage_history             % 损害历史
        radi_history               % RADI历史
        detection_history          % 检测历史
        reward_history             % 奖励历史
        
        %% === 新增：增强指标历史记录 ===
        nash_convergence_history     % Nash均衡收敛度历史
        attack_coverage_history      % 攻击覆盖率历史  
        defense_effectiveness_history % 防御有效性历史
        strategy_change_history      % 策略变化历史[attack_change, defense_change]
        
        %% === 新增：策略跟踪 ===
        prev_attack_strategy         % 上一轮攻击策略
        prev_defense_strategy        % 上一轮防御策略
        curr_attack_strategy         % 当前攻击策略
        curr_defense_strategy        % 当前防御策略
        
        %% === 兼容性属性 ===
        defense_costs           % 防御成本
        defense_effectiveness   % 防御效果
        radi_defender          % 兼容性RADI
        radi_attacker          % 兼容性RADI
        defender_strategy      % 兼容性策略
        attacker_strategy      % 兼容性策略
        optimal_defender_strategy % 兼容性
        optimal_attacker_strategy % 兼容性
        deployment_history     % 兼容性历史
        damage_history_alt     % 兼容性历史
        attack_history         % 兼容性历史
        defense_history        % 兼容性历史
        attacker_actions       % 兼容性
        defender_actions       % 兼容性
        
        %% === 其他参数 ===
        epsilon               % 数值稳定性参数
        debug_mode           % 调试模式
        optimization_method  % 优化方法（兼容性）
    end
    
    properties (Access = private)
        %% === 私有辅助属性 ===
        action_templates     % 预定义动作模板
        data_validation_enabled % 数据验证开关
    end
    
    methods (Access = public)
        function obj = TCSEnvironment(config)
            %TCSENVIRONMENT 构造函数（增强版）
            
            if nargin < 1
                error('TCSEnvironment:InvalidInput', '需要配置参数config');
            end
            
            % 验证配置
            obj.validateConfig(config);
            
            % 初始化基础参数
            obj.initializeBasicParameters(config);
            
            % 初始化环境组件
            obj.initializeEnvironmentComponents(config);
            
            % 初始化增强功能
            obj.initializeEnhancedFeatures(config);
            
            % 初始化历史记录（包括新增记录）
            obj.initializeAllHistoryRecords();
            
            % 计算空间维度
            obj.calculateSpaceDimensions();
            
            % 重置到初始状态
            obj.reset();
            
            if obj.debug_mode
                fprintf('[TCSEnvironment v4.0] 初始化完成 - %d站点, %d资源, 增强数据记录已启用\n', ...
                        obj.n_stations, obj.total_resources);
            end
        end
        
        function state = reset(obj)
            %RESET 重置环境到初始状态（增强版）
            
            obj.time_step = 0;
            
            % 重置FSP平均策略为均匀分布
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 重置内部Q表（如果使用）
            if ~isempty(obj.attacker_Q_table)
                obj.attacker_Q_table = randn(obj.max_defense_states, obj.n_stations) * 0.1;
            end
            
            % 重置兼容性策略
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.attacker_strategy = obj.station_values / sum(obj.station_values); % 基于价值初始化
            
            % 重置策略跟踪
            obj.resetStrategyTracking();
            
            % 清空所有历史记录
            obj.clearAllHistoryRecords();
            
            % 生成初始状态
            obj.current_state = obj.generateEnvironmentState();
            state = obj.current_state;
            
            if obj.debug_mode
                fprintf('[TCSEnvironment] 环境已重置，增强数据记录已初始化\n');
            end
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_deployment, attacker_target)
            %STEP 执行一步环境交互（增强版）
            
            % === 输入验证 ===
            obj.validateStepInputs(defender_deployment, attacker_target);
            
            % === 1. 更新FSP平均策略（环境维护的共享信息） ===
            obj.updateAttackerAverageStrategy(attacker_target);
            
            % === 2. 计算攻击结果 ===
            [attack_success, damage] = obj.computeAttackOutcome(attacker_target, defender_deployment);
            
            % === 3. 执行检测评估 ===
            detection_result = obj.evaluateDetection(attacker_target, defender_deployment, attack_success);
            
            % === 4. 计算奖励 ===
            [reward_def, reward_att] = obj.computeRewards(attack_success, damage, attacker_target, ...
                                                         defender_deployment, detection_result);
            
            % === 5. 更新环境状态（包括新增指标） ===
            obj.updateEnvironmentStateEnhanced(attack_success, damage, attacker_target, ...
                                              defender_deployment, detection_result);
            
            % === 6. 生成下一状态 ===
            next_state = obj.generateEnvironmentState();
            obj.current_state = next_state;
            obj.time_step = obj.time_step + 1;
            
            % === 7. 创建增强的信息结构 ===
            info = obj.createEnhancedStepInfo(attack_success, damage, attacker_target, ...
                                            defender_deployment, detection_result, reward_def, reward_att);
            
            % === 8. 数据验证（可选） ===
            if obj.data_validation_enabled
                obj.validateDataIntegrity();
            end
        end
        
        %% ========== 新增：策略管理方法 ==========
        
        function updateStrategies(obj, attack_strategy, defense_strategy)
            % 更新策略并记录变化
            % 输入：
            %   attack_strategy - 当前攻击策略向量
            %   defense_strategy - 当前防御策略向量
            
            % 验证输入
            if length(attack_strategy) ~= obj.n_stations || length(defense_strategy) ~= obj.n_stations
                warning('TCSEnvironment:InvalidStrategy', '策略向量长度与站点数不匹配');
                return;
            end
            
            % 保存上一轮策略
            obj.prev_attack_strategy = obj.curr_attack_strategy;
            obj.prev_defense_strategy = obj.curr_defense_strategy;
            
            % 更新当前策略
            obj.curr_attack_strategy = attack_strategy(:)'; % 确保行向量
            obj.curr_defense_strategy = defense_strategy(:)';
            
            % 归一化策略（确保概率和为1）
            if sum(obj.curr_attack_strategy) > 0
                obj.curr_attack_strategy = obj.curr_attack_strategy / sum(obj.curr_attack_strategy);
            end
            if sum(obj.curr_defense_strategy) > 0
                obj.curr_defense_strategy = obj.curr_defense_strategy / sum(obj.curr_defense_strategy);
            end
            
            if obj.debug_mode && mod(obj.time_step, 100) == 0
                fprintf('[策略更新] 时步%d: 攻击策略=%.3f, 防御策略=%.3f\n', ...
                        obj.time_step, obj.curr_attack_strategy(1), obj.curr_defense_strategy(1));
            end
        end
        
        function nash_convergence = calculateNashConvergence(obj)
            % 计算Nash均衡收敛度
            % 输出：nash_convergence - 收敛度（越小表示越接近Nash均衡）
            
            if isempty(obj.prev_attack_strategy) || isempty(obj.prev_defense_strategy) || ...
               isempty(obj.curr_attack_strategy) || isempty(obj.curr_defense_strategy)
                nash_convergence = 1.0; % 初始值
                return;
            end
            
            % 计算策略变化的L2范数
            attack_change = norm(obj.curr_attack_strategy - obj.prev_attack_strategy, 2);
            defense_change = norm(obj.curr_defense_strategy - obj.prev_defense_strategy, 2);
            
            % 综合收敛度（策略变化越小，收敛度越小）
            nash_convergence = (attack_change + defense_change) / 2;
            
            % 数值稳定性检查
            if isnan(nash_convergence) || isinf(nash_convergence)
                nash_convergence = 0;
            end
            
            % 限制在合理范围内
            nash_convergence = min(nash_convergence, 2.0); % 最大收敛度限制
        end
        
        function attack_coverage = calculateAttackCoverage(obj, defense_deployment, attack_success, detection_result)
            % 计算攻击覆盖率（防御系统能有效防御的攻击类型比例）
            % 输入：
            %   defense_deployment - 防御部署向量
            %   attack_success - 当前攻击是否成功
            %   detection_result - 检测结果结构体
            % 输出：
            %   attack_coverage - 攻击覆盖率 [0,1]
            
            % 防御强度（归一化的防御部署）
            if sum(defense_deployment) > 0
                defense_strength = defense_deployment / sum(defense_deployment);
            else
                defense_strength = zeros(size(defense_deployment));
            end
            
            % 基于检测能力和防御强度计算覆盖率
            detection_bonus = 0;
            if isfield(detection_result, 'detected') && detection_result.detected
                detection_bonus = 0.3; % 检测成功加分
            end
            
            defense_effectiveness = sum(defense_strength .* obj.station_values); % 防御有效性
            attack_failure_bonus = double(~attack_success) * 0.2; % 攻击失败加分
            
            % 基础覆盖率（基于防御部署的均匀性）
            defense_balance = 1 - std(defense_strength); % 部署越均匀，覆盖率越高
            
            % 综合覆盖率计算
            attack_coverage = min(0.95, max(0.05, ...
                defense_effectiveness * 0.4 + detection_bonus + attack_failure_bonus + defense_balance * 0.1));
            
            % 数值稳定性检查
            if isnan(attack_coverage) || isinf(attack_coverage)
                attack_coverage = 0.5; % 默认中等覆盖率
            end
        end
        
        function defense_effectiveness = calculateDefenseEffectiveness(obj, defense_deployment, damage, attack_success)
            % 计算防御有效性
            % 输入：
            %   defense_deployment - 防御部署
            %   damage - 造成的损害
            %   attack_success - 攻击是否成功
            % 输出：
            %   defense_effectiveness - 防御有效性 [0,1]
            
            max_possible_damage = max(obj.station_values);
            if max_possible_damage > 0
                damage_prevention = max(0, (max_possible_damage - damage) / max_possible_damage);
            else
                damage_prevention = 0;
            end
            
            attack_prevention = double(~attack_success);
            
            % 防御资源利用效率
            if obj.total_resources > 0
                resource_efficiency = sum(defense_deployment) / obj.total_resources;
            else
                resource_efficiency = 0;
            end
            
            % 综合防御有效性
            defense_effectiveness = 0.5 * damage_prevention + 0.3 * attack_prevention + 0.2 * resource_efficiency;
            
            % 限制在[0,1]范围内
            defense_effectiveness = max(0, min(1, defense_effectiveness));
            
            % 数值稳定性检查
            if isnan(defense_effectiveness) || isinf(defense_effectiveness)
                defense_effectiveness = 0.5;
            end
        end
        
        %% ========== 增强的历史记录管理 ==========
        
        function initializeAllHistoryRecords(obj)
            %INITIALIZEALLHISTORYRECORDS 初始化所有历史记录（包括新增）
            
            % 原有历史记录
            obj.attack_success_history = [];
            obj.attack_target_history = [];
            obj.defense_deployment_history = [];
            obj.damage_history = [];
            obj.radi_history = [];
            obj.detection_history = [];
            obj.reward_history = struct('defender', [], 'attacker', []);
            
            % 兼容性历史
            obj.deployment_history = [];
            obj.damage_history_alt = [];
            obj.attack_history = [];
            obj.defense_history = [];
            
            % 新增：增强指标历史记录
            obj.nash_convergence_history = [];
            obj.attack_coverage_history = [];
            obj.defense_effectiveness_history = [];
            obj.strategy_change_history = [];
            
            if obj.debug_mode
                fprintf('[历史记录] 所有历史记录已初始化\n');
            end
        end
        
        function clearAllHistoryRecords(obj)
            %CLEARALLHISTORYRECORDS 清空所有历史记录
            
            % 原有历史记录
            obj.attack_success_history = [];
            obj.attack_target_history = [];
            obj.defense_deployment_history = [];
            obj.damage_history = [];
            obj.radi_history = [];
            obj.detection_history = [];
            obj.reward_history = struct('defender', [], 'attacker', []);
            
            % 清空兼容性历史
            obj.deployment_history = [];
            obj.damage_history_alt = [];
            obj.attack_history = [];
            obj.defense_history = [];
            
            % 新增：清空增强指标历史
            obj.nash_convergence_history = [];
            obj.attack_coverage_history = [];
            obj.defense_effectiveness_history = [];
            obj.strategy_change_history = [];
        end
        
        function resetStrategyTracking(obj)
            %RESETSTRATEGYTRACKING 重置策略跟踪
            
            obj.prev_attack_strategy = [];
            obj.prev_defense_strategy = [];
            obj.curr_attack_strategy = [];
            obj.curr_defense_strategy = [];
        end
        
        function updateEnvironmentStateEnhanced(obj, attack_success, damage, attacker_target, defender_deployment, detection_result)
            %UPDATEENVIRONMENTSTATEENHANCED 更新环境状态（增强版）
            
            % === 原有记录（保持不变） ===
            obj.attack_success_history(end+1) = double(attack_success);
            obj.damage_history(end+1) = damage;
            obj.radi_history(end+1) = obj.radi_score;
            obj.detection_history(end+1) = detection_result.detected;
            
            % 记录攻击目标（one-hot编码）
            target_vector = zeros(1, obj.n_stations);
            target_vector(attacker_target) = 1;
            
            if isempty(obj.attack_target_history)
                obj.attack_target_history = target_vector;
                obj.defense_deployment_history = defender_deployment;
            else
                obj.attack_target_history(end+1, :) = target_vector;
                obj.defense_deployment_history(end+1, :) = defender_deployment;
            end
            
            % 更新兼容性历史
            obj.deployment_history = obj.defense_deployment_history;
            obj.damage_history_alt = obj.damage_history;
            obj.attack_history(end+1) = attacker_target;
            obj.defense_history = obj.defense_deployment_history;
            
            % === 新增：记录增强指标 ===
            
            % 1. Nash均衡收敛度记录
            nash_conv = obj.calculateNashConvergence();
            obj.nash_convergence_history(end+1) = nash_conv;
            
            % 2. 攻击覆盖率记录
            coverage = obj.calculateAttackCoverage(defender_deployment, attack_success, detection_result);
            obj.attack_coverage_history(end+1) = coverage;
            
            % 3. 防御有效性记录
            effectiveness = obj.calculateDefenseEffectiveness(defender_deployment, damage, attack_success);
            obj.defense_effectiveness_history(end+1) = effectiveness;
            
            % 4. 策略变化记录
            if ~isempty(obj.prev_attack_strategy) && ~isempty(obj.prev_defense_strategy)
                attack_change = norm(obj.curr_attack_strategy - obj.prev_attack_strategy, 2);
                defense_change = norm(obj.curr_defense_strategy - obj.prev_defense_strategy, 2);
                obj.strategy_change_history(end+1, :) = [attack_change, defense_change];
            else
                obj.strategy_change_history(end+1, :) = [0, 0];
            end
            
            % 更新兼容性策略和RADI
            obj.radi_defender = obj.radi_score;
            obj.radi_attacker = 0;
            
            % 更新当前策略（兼容性）
            if sum(defender_deployment) > 0
                obj.defender_strategy = defender_deployment / sum(defender_deployment);
            end
            
            % 更新攻击者策略（基于最近行为）
            recent_window = min(10, length(obj.attack_history));
            if recent_window > 0
                recent_targets = obj.attack_history(end-recent_window+1:end);
                strategy_update = zeros(1, obj.n_stations);
                for target = recent_targets
                    strategy_update(target) = strategy_update(target) + 1;
                end
                if sum(strategy_update) > 0
                    obj.attacker_strategy = strategy_update / sum(strategy_update);
                end
            end
        end
        
        function info = createEnhancedStepInfo(obj, attack_success, damage, attacker_target, ...
                                             defender_deployment, detection_result, reward_def, reward_att)
            % 创建增强的步骤信息结构体
            
            info = struct();
            
            % 基本信息
            info.attack_success = attack_success;
            info.damage = damage;
            info.attacker_target = attacker_target;
            info.defender_deployment = defender_deployment;
            info.time_step = obj.time_step;
            
            % 性能指标
            info.radi_score = obj.radi_score;
            info.detection_result = detection_result;
            
            % 奖励信息
            info.reward_def = reward_def;
            info.reward_att = reward_att;
            
            % 统计信息
            info.recent_success_rate = obj.computeRecentSuccessRate();
            info.recent_detection_rate = obj.computeRecentDetectionRate();
            info.recent_radi = obj.computeRecentRADI();
            
            % 新增：增强指标
            if ~isempty(obj.nash_convergence_history)
                info.current_nash_convergence = obj.nash_convergence_history(end);
            else
                info.current_nash_convergence = 1.0;
            end
            
            if ~isempty(obj.attack_coverage_history)
                info.current_attack_coverage = obj.attack_coverage_history(end);
            else
                info.current_attack_coverage = 0.5;
            end
            
            if ~isempty(obj.defense_effectiveness_history)
                info.current_defense_effectiveness = obj.defense_effectiveness_history(end);
            else
                info.current_defense_effectiveness = 0.5;
            end
            
            % 资源分配信息（用于性能监控）
            info.resource_allocation = defender_deployment / sum(defender_deployment);
            
            % 策略信息
            info.current_attack_strategy = obj.curr_attack_strategy;
            info.current_defense_strategy = obj.curr_defense_strategy;
        end
        
        %% ========== 数据验证方法 ==========
        
        function validateDataIntegrity(obj)
            % 验证数据完整性
            
            if ~obj.data_validation_enabled
                return;
            end
            
            % 检查历史数据长度一致性
            if ~isempty(obj.radi_history)
                expected_length = length(obj.radi_history);
                
                % 检查各个历史记录的长度
                history_fields = {
                    'nash_convergence_history',
                    'attack_coverage_history',
                    'defense_effectiveness_history'
                };
                
                for i = 1:length(history_fields)
                    field = history_fields{i};
                    if length(obj.(field)) ~= expected_length
                        warning('TCSEnvironment:DataInconsistency', ...
                                '历史数据长度不一致: %s (%d) vs expected (%d)', ...
                                field, length(obj.(field)), expected_length);
                    end
                end
            end
            
            % 检查数据范围合理性
            if ~isempty(obj.attack_coverage_history)
                coverage_data = obj.attack_coverage_history;
                if any(coverage_data < 0) || any(coverage_data > 1)
                    warning('TCSEnvironment:DataOutOfRange', '攻击覆盖率数据超出合理范围[0,1]');
                end
            end
            
            if ~isempty(obj.nash_convergence_history)
                nash_data = obj.nash_convergence_history;
                if any(nash_data < 0) || any(nash_data > 5)
                    warning('TCSEnvironment:DataOutOfRange', 'Nash收敛度数据超出合理范围[0,5]');
                end
            end
        end
        
        %% ========== 原有方法（保持兼容性） ==========
        
        function updateAttackerAverageStrategy(obj, attacker_target)
            %UPDATEATTACKERAVERAGESTY 更新攻击者平均策略（FSP）
            
            target_onehot = zeros(1, obj.n_stations);
            target_onehot(attacker_target) = 1;
            
            obj.attacker_avg_strategy = (1 - obj.alpha_ewma) * obj.attacker_avg_strategy + ...
                                       obj.alpha_ewma * target_onehot;
        end
        
        function [attack_success, damage] = computeAttackOutcome(obj, attacker_target, defender_deployment)
            %COMPUTEATTACKOUTCOME 计算攻击结果
            
            target_defense = defender_deployment(attacker_target);
            target_value = obj.station_values(attacker_target);
            
            % 攻击成功概率（基于防御强度）
            if obj.total_resources > 0
                defense_ratio = target_defense / obj.total_resources;
            else
                defense_ratio = 0;
            end
            
            attack_success_prob = max(0.1, min(0.9, 1 - defense_ratio));
            attack_success = rand() < attack_success_prob;
            
            % 计算损害
            if attack_success
                base_damage = target_value;
                defense_mitigation = defense_ratio * 0.5;
                damage = base_damage * (1 - defense_mitigation);
            else
                damage = 0;
            end
            
            damage = max(0, damage);
        end
        
        function detection_result = evaluateDetection(obj, attacker_target, defender_deployment, attack_success)
            %EVALUATEDETECTION 评估检测结果
            
            detection_result = struct();
            
            if ~obj.detection_enabled
                detection_result.detected = false;
                detection_result.detection_prob = 0;
                detection_result.is_false_positive = false;
                return;
            end
            
            % 计算检测概率
            target_defense = defender_deployment(attacker_target);
            if obj.total_resources > 0
                defense_factor = target_defense / obj.total_resources;
            else
                defense_factor = 0;
            end
            
            detection_prob = obj.base_detection_rate + defense_factor * obj.detection_sensitivity;
            detection_prob = min(0.95, detection_prob);
            
            % 确定检测结果
            if attack_success
                detected = rand() < detection_prob;
                is_false_positive = false;
            else
                detected = rand() < obj.false_positive_rate;
                is_false_positive = detected;
            end
            
            detection_result.detected = detected;
            detection_result.detection_prob = detection_prob;
            detection_result.is_false_positive = is_false_positive;
        end
        
        function [reward_def, reward_att] = computeRewards(obj, attack_success, damage, attacker_target, ...
                                                          defender_deployment, detection_result)
            %COMPUTEREWARDS 计算奖励
            
            % 攻击者奖励
            if attack_success
                reward_att = damage;
                if detection_result.detected && ~detection_result.is_false_positive
                    reward_att = reward_att * 0.5; % 被检测到时奖励减半
                end
            else
                reward_att = -0.1; % 攻击失败的小惩罚
            end
            
            % 防御者奖励
            reward_def = -damage; % 基础：避免损害
            
            if detection_result.detected && ~detection_result.is_false_positive
                reward_def = reward_def + 0.5; % 成功检测奖励
            elseif detection_result.is_false_positive
                reward_def = reward_def - 0.2; % 误报惩罚
            end
            
            % 防御效率奖励
            if obj.total_resources > 0
                resource_efficiency = 1 - (sum(defender_deployment) / obj.total_resources);
                reward_def = reward_def + resource_efficiency * 0.1;
            end
        end
        
        function radi = calculateRADI(obj, defender_deployment)
            %CALCULATERADI 计算RADI指标
            
            if sum(defender_deployment) == 0
                radi = 1.0; % 最差情况
                return;
            end
            
            % 归一化部署
            normalized_deployment = defender_deployment / sum(defender_deployment);
            
            % 获取最优配置
            if isfield(obj.radi_config, 'optimal_allocation')
                optimal_allocation = obj.radi_config.optimal_allocation;
            else
                optimal_allocation = ones(1, obj.n_stations) / obj.n_stations;
            end
            
            % 获取权重
            if isfield(obj.radi_config, 'weights')
                weights = obj.radi_config.weights;
            else
                weights = obj.station_values / sum(obj.station_values);
            end
            
            % 计算偏差
            deviation = abs(normalized_deployment - optimal_allocation);
            weighted_deviation = sum(weights .* deviation);
            
            % RADI分数（0表示完美，1表示最差）
            radi = min(1.0, weighted_deviation);
            
            % 更新当前RADI分数
            obj.radi_score = radi;
            
            % 数值稳定性检查
            if isnan(radi) || isinf(radi)
                radi = 0;
            end
        end
        
        function success_rate = computeRecentSuccessRate(obj)
            %COMPUTERECENTSUCCESE 计算最近攻击成功率
            
            if isempty(obj.attack_success_history)
                success_rate = 0;
                return;
            end
            
            recent_window = min(50, length(obj.attack_success_history));
            recent_data = obj.attack_success_history(end-recent_window+1:end);
            success_rate = mean(recent_data);
        end
        
        function detection_rate = computeRecentDetectionRate(obj)
            %COMPUTERECENTDETECTIONRATE 计算最近检测率
            
            if isempty(obj.detection_history)
                detection_rate = 0;
                return;
            end
            
            recent_window = min(50, length(obj.detection_history));
            recent_data = obj.detection_history(end-recent_window+1:end);
            detection_rate = mean(recent_data);
        end
        
        function recent_radi = computeRecentRADI(obj)
            %COMPUTERECENTRADI 计算最近RADI
            
            if isempty(obj.radi_history)
                recent_radi = 0;
                return;
            end
            
            recent_window = min(20, length(obj.radi_history));
            recent_data = obj.radi_history(end-recent_window+1:end);
            recent_radi = mean(recent_data);
        end
        
        function state = generateEnvironmentState(obj)
            %GENERATEENVIRONMENTSTATE 生成环境状态向量
            
            % 状态组成：[攻击者平均策略, 时间归一化, 最近RADI]
            time_norm = min(obj.time_step / 1000, 1.0);
            recent_radi = obj.computeRecentRADI();
            
            state = [obj.attacker_avg_strategy, time_norm, recent_radi];
            
            % 确保状态长度正确
            if length(state) ~= obj.state_dim
                if length(state) < obj.state_dim
                    state = [state, zeros(1, obj.state_dim - length(state))];
                else
                    state = state(1:obj.state_dim);
                end
            end
        end
        
        %% ========== 辅助和兼容性方法 ==========
        
        function deployment = parseDefenderAction(obj, action)
            %PARSEDEFENDERACTION 解析防御者动作（兼容性）
            
            if isscalar(action)
                % 使用预定义模板
                template_idx = min(max(round(action), 1), size(obj.action_templates, 1));
                deployment = obj.action_templates(template_idx, :) * obj.total_resources;
            elseif length(action) == obj.n_stations
                deployment = action;
                if sum(deployment) > 0
                    deployment = deployment / sum(deployment) * obj.total_resources;
                end
            else
                deployment = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
            end
            
            deployment = max(0, deployment);
        end
        
        function target = parseAttackerAction(obj, action)
            %PARSEATTACKERACTION 解析攻击者动作（兼容性）
            
            if isscalar(action)
                target = min(max(round(action), 1), obj.n_stations);
            else
                [~, target] = max(action);
            end
        end
        
        function info = getEnvironmentInfo(obj)
            %GETENVIRONMENTINFO 获取环境信息
            
            info = struct();
            info.n_stations = obj.n_stations;
            info.total_resources = obj.total_resources;
            info.time_step = obj.time_step;
            info.current_radi = obj.radi_score;
            
            % 性能统计
            info.recent_success_rate = obj.computeRecentSuccessRate();
            info.recent_detection_rate = obj.computeRecentDetectionRate();
            info.recent_radi = obj.computeRecentRADI();
            
            % 新增：增强指标统计
            if ~isempty(obj.nash_convergence_history)
                info.recent_nash_convergence = mean(obj.nash_convergence_history(max(1, end-19):end));
            else
                info.recent_nash_convergence = 1.0;
            end
            
            if ~isempty(obj.attack_coverage_history)
                info.recent_attack_coverage = mean(obj.attack_coverage_history(max(1, end-19):end));
            else
                info.recent_attack_coverage = 0.5;
            end
            
            if ~isempty(obj.defense_effectiveness_history)
                info.recent_defense_effectiveness = mean(obj.defense_effectiveness_history(max(1, end-19):end));
            else
                info.recent_defense_effectiveness = 0.5;
            end
        end
        
        function summary = getDataSummary(obj)
            % 获取数据摘要（用于调试和监控）
            
            summary = struct();
            summary.total_steps = obj.time_step;
            
            % 原有指标
            summary.radi_data_points = length(obj.radi_history);
            summary.attack_success_data_points = length(obj.attack_success_history);
            
            % 新增指标
            summary.nash_convergence_data_points = length(obj.nash_convergence_history);
            summary.attack_coverage_data_points = length(obj.attack_coverage_history);
            summary.defense_effectiveness_data_points = length(obj.defense_effectiveness_history);
            summary.strategy_change_data_points = size(obj.strategy_change_history, 1);
            
            % 最终值
            if ~isempty(obj.radi_history)
                summary.final_radi = obj.radi_history(end);
            else
                summary.final_radi = NaN;
            end
            
            if ~isempty(obj.nash_convergence_history)
                summary.final_nash_convergence = obj.nash_convergence_history(end);
            else
                summary.final_nash_convergence = NaN;
            end
            
            if ~isempty(obj.attack_coverage_history)
                summary.final_attack_coverage = obj.attack_coverage_history(end);
            else
                summary.final_attack_coverage = NaN;
            end
        end
    end
    
    methods (Access = private)
        function validateConfig(obj, config)
            %VALIDATECONFIG 验证配置参数
            
            if ~isfield(config, 'n_stations') || config.n_stations <= 0
                error('TCSEnvironment:InvalidConfig', '需要有效的n_stations参数');
            end
            
            if isfield(config, 'n_components_per_station')
                if length(config.n_components_per_station) ~= config.n_stations
                    error('TCSEnvironment:InvalidConfig', 'n_components_per_station长度必须等于n_stations');
                end
            end
        end
        
        function initializeBasicParameters(obj, config)
            %INITIALIZEBASICPARAMETERS 初始化基础参数
            
            % 基础系统参数
            obj.n_stations = config.n_stations;
            
            % 组件配置
            if isfield(config, 'n_components_per_station') && ~isempty(config.n_components_per_station)
                obj.n_components = config.n_components_per_station(:)';
                if length(obj.n_components) ~= obj.n_stations
                    obj.n_components = repmat(3, 1, obj.n_stations);
                end
            else
                obj.n_components = repmat(3, 1, obj.n_stations);
            end
            obj.total_components = sum(obj.n_components);
            
            % 资源配置
            obj.total_resources = obj.getConfigValue(config, 'total_resources', 100);
            obj.n_resource_types = obj.getConfigValue(config, 'n_resource_types', 5);
            obj.n_attack_types = obj.getConfigValue(config, 'n_attack_types', 6);
            
            % 生成站点价值
            obj.generateStationValues();
            
            % 其他参数
            obj.epsilon = 1e-8;
            obj.debug_mode = obj.getConfigValue(config, 'debug_mode', false);
            obj.optimization_method = 'default';
        end
        
        function initializeEnvironmentComponents(obj, config)
            %INITIALIZEENVIRONMENTCOMPONENTS 初始化环境组件
            
            % FSP参数
            obj.alpha_ewma = obj.getConfigValue(config, 'alpha_ewma', 0.1);
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 内部Q-Learning参数
            obj.attacker_lr = obj.getConfigValue(config, 'attacker_lr', 0.1);
            obj.attacker_gamma = obj.getConfigValue(config, 'attacker_gamma', 0.95);
            obj.attacker_epsilon = obj.getConfigValue(config, 'attacker_epsilon', 0.3);
            obj.attacker_epsilon_decay = obj.getConfigValue(config, 'attacker_epsilon_decay', 0.995);
            obj.attacker_epsilon_min = obj.getConfigValue(config, 'attacker_epsilon_min', 0.01);
            obj.max_defense_states = 50;
            obj.attacker_Q_table = randn(obj.max_defense_states, obj.n_stations) * 0.1;
            
            % RADI配置
            obj.radi_config = struct();
            if isfield(config, 'radi')
                obj.radi_config = config.radi;
            else
                obj.radi_config.optimal_allocation = ones(1, obj.n_stations) / obj.n_stations;
                obj.radi_config.weights = ones(1, obj.n_stations) / obj.n_stations;
            end
            
            % 检测系统
            obj.detection_enabled = obj.getConfigValue(config, 'detection_enabled', true);
            obj.base_detection_rate = obj.getConfigValue(config, 'base_detection_rate', 0.3);
            obj.detection_sensitivity = obj.getConfigValue(config, 'detection_sensitivity', 0.8);
            obj.false_positive_rate = obj.getConfigValue(config, 'false_positive_rate', 0.1);
            
            % 创建动作模板
            obj.action_templates = obj.createActionTemplates();
            
            % 兼容性初始化
            obj.initializeCompatibilityAttributes();
        end
        
        function initializeEnhancedFeatures(obj, config)
            %INITIALIZEENHANCEDFEATURES 初始化增强功能
            
            % 数据验证开关
            obj.data_validation_enabled = obj.getConfigValue(config, 'enable_data_validation', true);
            
            % 初始化策略跟踪
            obj.resetStrategyTracking();
            
            if obj.debug_mode
                fprintf('[增强功能] 数据验证: %s\n', ...
                        obj.data_validation_enabled ? '启用' : '禁用');
            end
        end
        
        function initializeCompatibilityAttributes(obj)
            %INITIALIZECOMPATIBILITYATTRIBUTES 初始化兼容性属性
            
            obj.defense_costs = ones(1, obj.n_stations);
            obj.defense_effectiveness = ones(1, obj.n_stations) * 0.8;
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.attacker_strategy = obj.station_values / sum(obj.station_values);
            obj.optimal_defender_strategy = obj.defender_strategy;
            obj.optimal_attacker_strategy = obj.attacker_strategy;
            obj.attacker_actions = [];
            obj.defender_actions = [];
        end
        
        function calculateSpaceDimensions(obj)
            %CALCULATESPACEDIMENSIONS 计算状态和动作空间维度
            
            % 状态维度：[攻击者平均策略, 时间归一化, RADI]
            obj.state_dim = obj.n_stations + 1 + 1;
            
            % 动作维度
            obj.action_dim = obj.n_stations;
            obj.action_dim_defender = obj.n_stations;
            obj.action_dim_attacker = obj.n_stations;
        end
        
        function generateStationValues(obj)
            %GENERATESTATIONVALUES 生成站点价值
            
            % 基于组件数量生成价值
            component_importance = rand(1, obj.total_components);
            obj.station_values = zeros(1, obj.n_stations);
            
            idx = 1;
            for i = 1:obj.n_stations
                n_comp = obj.n_components(i);
                obj.station_values(i) = sum(component_importance(idx:idx+n_comp-1));
                idx = idx + n_comp;
            end
            
            % 归一化并增加差异性
            obj.station_values = obj.station_values / sum(obj.station_values);
            obj.station_values = obj.station_values .^ 0.8; % 减少极端差异
            obj.station_values = obj.station_values / sum(obj.station_values);
        end
        
        function validateStepInputs(obj, defender_deployment, attacker_target)
            %VALIDATESTEPINPUTS 验证step方法输入
            
            if length(defender_deployment) ~= obj.n_stations
                error('TCSEnvironment:InvalidInput', '防御部署向量长度必须为%d', obj.n_stations);
            end
            
            if ~isscalar(attacker_target) || attacker_target < 1 || attacker_target > obj.n_stations
                error('TCSEnvironment:InvalidInput', '攻击目标必须在1到%d之间', obj.n_stations);
            end
            
            if any(defender_deployment < 0)
                error('TCSEnvironment:InvalidInput', '防御部署不能为负数');
            end
        end
        
        function templates = createActionTemplates(obj)
            %CREATEACTIONTEMPLATES 创建动作模板
            
            n_templates = 5;
            templates = zeros(n_templates, obj.n_stations);
            
            % 均匀分配
            templates(1, :) = ones(1, obj.n_stations) / obj.n_stations;
            
            % 基于价值的分配
            templates(2, :) = obj.station_values / sum(obj.station_values);
            
            % 随机分配
            for i = 3:n_templates
                random_weights = rand(1, obj.n_stations);
                templates(i, :) = random_weights / sum(random_weights);
            end
        end
        
        function value = getConfigValue(obj, config, field, default_value)
            %GETCONFIGVALUE 安全获取配置值
            
            if isfield(config, field)
                value = config.(field);
            else
                value = default_value;
            end
        end
    end
    
    methods (Static)
        function config = getDefaultConfig()
            %GETDEFAULTCONFIG 获取默认配置
            
            config = struct();
            
            % === 基础系统参数 ===
            config.n_stations = 5;
            config.n_components_per_station = [3, 3, 3, 3, 3];
            config.total_resources = 100;
            config.n_resource_types = 5;
            config.n_attack_types = 6;
            
            % === FSP参数 ===
            config.alpha_ewma = 0.1;
            
            % === 内部Q-Learning参数 ===
            config.attacker_lr = 0.1;
            config.attacker_gamma = 0.95;
            config.attacker_epsilon = 0.3;
            config.attacker_epsilon_decay = 0.995;
            config.attacker_epsilon_min = 0.01;
            
            % === RADI配置 ===
            config.radi = struct();
            config.radi.optimal_allocation = ones(1, 5) / 5;
            config.radi.weights = ones(1, 5) / 5;
            
            % === 检测系统参数 ===
            config.detection_enabled = true;
            config.base_detection_rate = 0.3;
            config.detection_sensitivity = 0.8;
            config.false_positive_rate = 0.1;
            
            % === 增强功能参数 ===
            config.enable_data_validation = true;
            config.debug_mode = false;
        end
        
        function config = getOptimizedConfig()
            %GETOPTIMIZEDCONFIG 获取优化配置
            
            config = TCSEnvironment.getDefaultConfig();
            
            % === 性能优化参数 ===
            config.alpha_ewma = 0.05;
            config.attacker_lr = 0.03;
            config.attacker_epsilon = 0.5;
            config.attacker_epsilon_decay = 0.9995;
            config.attacker_epsilon_min = 0.1;
            
            % === 检测系统增强 ===
            config.base_detection_rate = 0.4;
            config.detection_sensitivity = 0.85;
            config.false_positive_rate = 0.08;
            
            % === 调试和监控 ===
            config.debug_mode = true;
            config.enable_data_validation = true;
        end
        
        function config = getTestConfig()
            %GETTESTCONFIG 获取测试配置
            
            config = TCSEnvironment.getDefaultConfig();
            
            % 小规模测试参数
            config.n_stations = 3;
            config.n_components_per_station = [2, 2, 2];
            config.total_resources = 50;
            config.debug_mode = true;
            
            % 快速收敛参数
            config.alpha_ewma = 0.2;
            config.attacker_lr = 0.2;
            config.attacker_epsilon_decay = 0.99;
            
            % 启用所有增强功能
            config.enable_data_validation = true;
        end
        
        function demo()
            %DEMO 演示TCSEnvironment v4.0的使用
            
            fprintf('=== TCSEnvironment v4.0 增强版演示 ===\n\n');
            
            % 创建配置
            config = TCSEnvironment.getTestConfig();
            
            % 创建环境
            env = TCSEnvironment(config);
            
            fprintf('1. 基础功能测试...\n');
            state = env.reset();
            fprintf('   初始状态维度: %d\n', length(state));
            
            fprintf('\n2. 策略记录测试...\n');
            attack_strategy = [0.4, 0.3, 0.3];
            defense_strategy = [0.2, 0.5, 0.3];
            env.updateStrategies(attack_strategy, defense_strategy);
            fprintf('   策略已更新\n');
            
            fprintf('\n3. 环境交互测试...\n');
            for i = 1:10
                defender_deployment = [15, 20, 15];
                attacker_target = randi(3);
                
                [next_state, reward_def, reward_att, info] = env.step(defender_deployment, attacker_target);
                
                if i == 1
                    fprintf('   步骤%d: RADI=%.3f, Nash收敛度=%.3f, 攻击覆盖率=%.3f\n', ...
                            i, info.radi_score, info.current_nash_convergence, info.current_attack_coverage);
                end
            end
            
            fprintf('\n4. 数据摘要:\n');
            summary = env.getDataSummary();
            fprintf('   总步数: %d\n', summary.total_steps);
            fprintf('   RADI数据点: %d\n', summary.radi_data_points);
            fprintf('   Nash收敛度数据点: %d\n', summary.nash_convergence_data_points);
            fprintf('   攻击覆盖率数据点: %d\n', summary.attack_coverage_data_points);
            
            fprintf('\n✅ TCSEnvironment v4.0 演示完成！\n');
        end
        
        function runPerformanceTest()
            %RUNPERFORMANCETEST 运行性能测试
            
            fprintf('=== TCSEnvironment v4.0 性能测试 ===\n\n');
            
            config = TCSEnvironment.getOptimizedConfig();
            config.debug_mode = false;
            
            fprintf('测试配置: %d站点, %d资源\n', config.n_stations, config.total_resources);
            
            env = TCSEnvironment(config);
            
            n_episodes = 50;
            n_steps_per_episode = 100;
            
            fprintf('开始性能测试: %d episodes × %d steps\n\n', n_episodes, n_steps_per_episode);
            
            total_start = tic;
            
            for episode = 1:n_episodes
                env.reset();
                
                for step = 1:n_steps_per_episode
                    % 智能策略
                    if step == 1
                        defender_deployment = ones(1, env.n_stations) * (env.total_resources / env.n_stations);
                    else
                        threat_weights = env.attacker_avg_strategy .* env.station_values;
                        if sum(threat_weights) > 0
                            defense_weights = threat_weights / sum(threat_weights);
                        else
                            defense_weights = ones(1, env.n_stations) / env.n_stations;
                        end
                        defender_deployment = defense_weights * env.total_resources;
                    end
                    
                    % 攻击策略
                    defense_strength = defender_deployment / env.total_resources;
                    attack_attractiveness = env.station_values ./ (defense_strength + 0.1);
                    [~, attacker_target] = max(attack_attractiveness);
                    
                    % 更新策略记录
                    attack_strategy = zeros(1, env.n_stations);
                    attack_strategy(attacker_target) = 1;
                    defense_strategy = defender_deployment / sum(defender_deployment);
                    env.updateStrategies(attack_strategy, defense_strategy);
                    
                    % 执行交互
                    [~, ~, ~, info] = env.step(defender_deployment, attacker_target);
                end
                
                if mod(episode, 10) == 0
                    fprintf('Episode %d/%d completed\n', episode, n_episodes);
                end
            end
            
            total_time = toc(total_start);
            
            fprintf('\n=== 性能测试结果 ===\n');
            fprintf('总时间: %.2f秒\n', total_time);
            fprintf('平均每步时间: %.4f秒\n', total_time / (n_episodes * n_steps_per_episode));
            
            summary = env.getDataSummary();
            fprintf('数据记录完整性:\n');
            fprintf('  RADI数据点: %d\n', summary.radi_data_points);
            fprintf('  Nash收敛度数据点: %d\n', summary.nash_convergence_data_points);
            fprintf('  攻击覆盖率数据点: %d\n', summary.attack_coverage_data_points);
            
            fprintf('\n✅ 性能测试完成！\n');
        end
    end
end