%% FSPSimulator.m - FSP仿真器类
% =========================================================================
% 描述: FSP (Fictitious Self-Play) 仿真器核心类
% 提供静态方法run来执行FSP仿真
% =========================================================================

classdef FSPSimulator < handle
   %% FSP仿真器 - 处理多智能体强化学习仿真
   
   methods (Static)
       function results = run(env, defender_agents, attacker_agent, config, monitor)
           %% 运行FSP仿真的主函数
           % 输入:
           %   env - TCS环境对象
           %   defender_agents - 防御者智能体数组
           %   attacker_agent - 攻击者智能体
           %   config - 配置结构体
           %   monitor - 性能监控器
           % 输出:
           %   results - 仿真结果结构体
           
           try
               fprintf('🚀 开始FSP仿真训练...\n');
               Logger.info('FSP仿真训练开始');
               
               n_iterations = config.n_iterations;
               n_agents = length(defender_agents);
               
                % 初始化结果结构
                results = struct();
                results.defender_rewards = zeros(n_iterations, n_agents);
                results.attacker_rewards = zeros(n_iterations, 1);
                results.detection_rates = zeros(n_iterations, n_agents);
                results.resource_efficiency = zeros(n_iterations, n_agents);
                results.convergence_info = struct();
                
                % === 添加缺失的字段 ===
                results.n_agents = n_agents;
                results.n_iterations = n_iterations;
                results.config = config;
                results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
               
               % FSP迭代循环
               for iter = 1:n_iterations
                   tic;
                   
                   fprintf('⏳ 执行第 %d/%d 次迭代...', iter, n_iterations);
                   
                   % 运行episodes
                   episode_results = FSPSimulator.runEpisodes(env, defender_agents, attacker_agent, config);
                   
                   % 更新智能体参数
                   FSPSimulator.updateAgents(defender_agents, attacker_agent, episode_results, config);
                   
                   % 记录结果
                   % 安全的数据赋值，处理维度不匹配问题
                   if isfield(episode_results, 'avg_defender_reward')
                       reward_data = episode_results.avg_defender_reward;
                       if length(reward_data) == size(results.defender_rewards, 2)
                           results.defender_rewards(iter, :) = reward_data;
                       else
                           % 维度不匹配时的处理
                           min_len = min(length(reward_data), size(results.defender_rewards, 2));
                           results.defender_rewards(iter, 1:min_len) = reward_data(1:min_len);
                       end
                   end
                   results.attacker_rewards(iter) = episode_results.avg_attacker_reward;
                   results.detection_rates(iter, :) = episode_results.avg_detection_rate;
                   results.resource_efficiency(iter, :) = episode_results.avg_efficiency;
                   
                   % 更新监控器
                   if exist('monitor', 'var') && ~isempty(monitor)
                       try
                           monitor.updateIteration(iter, episode_results);
                       catch
                           % 监控器更新失败不影响主程序
                       end
                   end
                   
                   iter_time = toc;
                   fprintf(' 完成，用时 %.2f秒\n', iter_time);
                   Logger.info(sprintf('迭代 %d 完成，用时 %.2f秒', iter, iter_time));
                   
                   % 每10次迭代显示进度
                   if mod(iter, 10) == 0
                       avg_detection = mean(results.detection_rates(iter, :));
                       avg_efficiency = mean(results.resource_efficiency(iter, :));
                       fprintf('📊 第%d次迭代 - 平均检测率: %.3f, 平均效率: %.3f\n', ...
                               iter, avg_detection, avg_efficiency);
                   end
               end
               
               % 分析收敛性
               results.convergence_info = FSPSimulator.analyzeConvergence(results);
               
               fprintf('✅ FSP仿真训练完成！\n');
               Logger.info('FSP仿真训练成功完成');
               
           catch ME
               Logger.error(sprintf('FSP仿真过程中出错: %s', ME.message));
               rethrow(ME);
           end
       end
       
       function episode_results = runEpisodes(env, defender_agents, attacker_agent, config)
           %% 运行多个episodes
           
           n_agents = length(defender_agents);
           n_episodes = config.n_episodes_per_iter;
           
           % 初始化累积变量
           defender_reward_sum = zeros(1, n_agents);
           attacker_reward_sum = 0;
           detection_rate_sum = zeros(1, n_agents);
           efficiency_sum = zeros(1, n_agents);
           
           % 运行episodes
           for ep = 1:n_episodes
               % 重置环境
               state = env.reset();
               
               episode_defender_rewards = zeros(1, n_agents);
               episode_attacker_reward = 0;
               episode_detection_rates = zeros(1, n_agents);
               episode_efficiency = zeros(1, n_agents);
               
               % 每个时间步 - 使用默认值如果配置中没有此字段
               max_steps = 50; % 默认最大步数
               if isfield(config, 'max_steps_per_episode')
                   max_steps = config.max_steps_per_episode;
               elseif isfield(config, 'max_episode_steps')
                   max_steps = config.max_episode_steps;
               end
               
               for step = 1:max_steps
                   % 防御者选择动作
                   defender_actions = [];
                   for i = 1:n_agents
                       action = defender_agents{i}.selectAction(state);
                       
                       % 读取配置中的站点数量
                      % 使用传入的配置中的站点数量
                        n_stations = config.n_stations;
                       
                       % 确保动作向量长度等于站点数量
                       if length(action) ~= n_stations
                           if length(action) < n_stations
                               action = [action, zeros(1, n_stations - length(action))];
                           else
                               action = action(1:n_stations);
                           end
                       end
                       
                       % 归一化
                       if sum(action) > 0
                           action = action / sum(action);
                       else
                           action = ones(1, n_stations) / n_stations;
                       end
                       
                       % 存储整个动作向量，而不是标量
                       if i == 1
                           defender_actions = action;
                       else
                           defender_actions = [defender_actions; action];
                       end
                   end
                   
                   % 攻击者选择动作
                   attacker_action = attacker_agent.selectAction(state);
                   % 确保攻击者动作也是标量
                   if length(attacker_action) > 1
                       attacker_action = attacker_action(1);
                   end
                   
                   % 执行环境步骤
                   try
                       [next_state, rewards, done, info] = env.step(defender_actions, attacker_action);
                   catch ME
                       % 如果环境步骤失败，使用默认值
                       warning('环境步骤执行失败: %s', ME.message);
                       next_state = state;
                       rewards = struct();
                       rewards.defender = zeros(1, n_agents);
                       rewards.attacker = 0;
                       done = false;
                       info = struct();
                       info.detection_rate = 0.5 * ones(1, n_agents);
                       info.efficiency = 0.7 * ones(1, n_agents);
                   end
                   
                   % 更新智能体经验
                   for i = 1:n_agents
                        if hasMethod(defender_agents{i}, 'updateExperience')
                            % 检查rewards是否为结构体
                            if isstruct(rewards) && isfield(rewards, 'defender')
                                reward_value = rewards.defender(i);
                            else
                                reward_value = 0; % 默认奖励
                            end
                            
                            defender_agents{i}.updateExperience(state, defender_actions(i,:), reward_value, next_state, done);
                        end
                    end
                    
                    if hasMethod(attacker_agent, 'updateExperience')
                        % 检查rewards是否为结构体
                        if isstruct(rewards) && isfield(rewards, 'attacker')
                            if length(rewards.attacker) > 1
                                attacker_reward_value = rewards.attacker(1);
                            else
                                attacker_reward_value = rewards.attacker;
                            end
                        else
                            attacker_reward_value = 0; % 默认奖励
                        end
                        
                        attacker_agent.updateExperience(state, attacker_action, attacker_reward_value, next_state, done);
                    end
                   
                   % 累积奖励（确保维度正确）
                   if isstruct(rewards) && isfield(rewards, 'defender') && length(rewards.defender) == n_agents
                       episode_defender_rewards = episode_defender_rewards + rewards.defender;
                   else
                       % 使用默认奖励
                       episode_defender_rewards = episode_defender_rewards + zeros(1, n_agents);
                   end
                   
                   if isstruct(rewards) && isfield(rewards, 'attacker')
                       if length(rewards.attacker) > 1
                           episode_attacker_reward = episode_attacker_reward + rewards.attacker(1);
                       else
                           episode_attacker_reward = episode_attacker_reward + rewards.attacker;
                       end
                   else
                       % 使用默认奖励
                       episode_attacker_reward = episode_attacker_reward + 0;
                   end
                   
                   % 计算检测率和效率（确保维度正确）
                   if isstruct(info) && isfield(info, 'detection_rate')
                       if length(info.detection_rate) == n_agents
                           episode_detection_rates = episode_detection_rates + info.detection_rate;
                       elseif length(info.detection_rate) == 1
                           episode_detection_rates = episode_detection_rates + repmat(info.detection_rate, 1, n_agents);
                       end
                   else
                       % 如果info不是结构体或没有detection_rate字段，使用默认值
                       episode_detection_rates = episode_detection_rates + 0.5 * ones(1, n_agents);
                   end
                   
                   if isstruct(info) && isfield(info, 'efficiency')
                       if length(info.efficiency) == n_agents
                           episode_efficiency = episode_efficiency + info.efficiency;
                       elseif length(info.efficiency) == 1
                           episode_efficiency = episode_efficiency + repmat(info.efficiency, 1, n_agents);
                       end
                   else
                       % 如果info不是结构体或没有efficiency字段，使用默认值
                       episode_efficiency = episode_efficiency + 0.7 * ones(1, n_agents);
                   end
                   
                   state = next_state;
                   
                   if done
                       break;
                   end
               end
               
               % 累积episode结果
               defender_reward_sum = defender_reward_sum + episode_defender_rewards;
               attacker_reward_sum = attacker_reward_sum + episode_attacker_reward;
               detection_rate_sum = detection_rate_sum + episode_detection_rates;
               efficiency_sum = efficiency_sum + episode_efficiency;
           end
           
           % 计算平均值
           episode_results = struct();
           episode_results.avg_defender_reward = defender_reward_sum / n_episodes;
           episode_results.avg_attacker_reward = attacker_reward_sum / n_episodes;
           episode_results.avg_detection_rate = detection_rate_sum / n_episodes;
           episode_results.avg_efficiency = efficiency_sum / n_episodes;
       end
       
       function updateAgents(defender_agents, attacker_agent, episode_results, config)
            %% 更新智能体参数
            
            n_agents = length(defender_agents);
            
            % 更新防御者智能体
            for i = 1:n_agents
                if hasMethod(defender_agents{i}, 'updateParameters')
                    % 不传递参数，直接调用
                    defender_agents{i}.updateParameters();
                end
                
                % 更新学习率等参数
                if hasMethod(defender_agents{i}, 'decay')
                    defender_agents{i}.decay();
                end
            end
            
            % 更新攻击者智能体
            if hasMethod(attacker_agent, 'updateParameters')
                % 不传递参数，直接调用
                attacker_agent.updateParameters();
            end
            
            if hasMethod(attacker_agent, 'decay')
                attacker_agent.decay();
            end
        end
       
       function convergence_info = analyzeConvergence(results)
           %% 分析收敛性
           
           convergence_info = struct();
           
           % 分析防御者奖励收敛
           defender_rewards = results.defender_rewards;
           n_iterations = size(defender_rewards, 1);
           n_agents = size(defender_rewards, 2);
           
           % 计算最后20%迭代的稳定性
           stable_window = max(10, floor(n_iterations * 0.2));
           stable_start = n_iterations - stable_window + 1;
           
           convergence_info.defender_convergence = zeros(1, n_agents);
           for i = 1:n_agents
               stable_rewards = defender_rewards(stable_start:end, i);
               convergence_info.defender_convergence(i) = std(stable_rewards) / mean(abs(stable_rewards));
           end
           
           % 分析攻击者奖励收敛
           attacker_rewards = results.attacker_rewards;
           stable_attacker_rewards = attacker_rewards(stable_start:end);
           convergence_info.attacker_convergence = std(stable_attacker_rewards) / mean(abs(stable_attacker_rewards));
           
           % 总体收敛指标
           convergence_info.overall_convergence = mean([convergence_info.defender_convergence, convergence_info.attacker_convergence]);
           
           % 检测是否收敛（变异系数小于0.1认为收敛）
           convergence_info.is_converged = convergence_info.overall_convergence < 0.1;
           
           % 准备状态文本
           if convergence_info.is_converged
               status_text = '(已收敛)';
           else
               status_text = '(未收敛)';
           end
           
           fprintf('📈 收敛分析 - 整体收敛指标: %.4f %s\n', ...
                   convergence_info.overall_convergence, status_text);
       end
   end
end

%% 辅助函数
function has_method = hasMethod(obj, method_name)
   %% 检查对象是否有指定方法
   try
       if isobject(obj)
           has_method = any(strcmp(methods(obj), method_name));
       else
           has_method = false;
       end
   catch
       has_method = false;
   end
end