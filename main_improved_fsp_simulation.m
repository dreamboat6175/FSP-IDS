%% main_fsp_cyber_battle.m - 基于FSP的网络攻防博弈仿真
% =========================================================================
% 实现理性攻击者和FSP防御者的博弈模型
% =========================================================================

clear all; close all; clc;

%% 1. 配置参数
config = struct();

% 环境参数
config.n_stations = 5;
config.n_components_per_station = [3, 4, 5, 4, 3];
config.n_resource_types = 5;
config.n_attack_types = 6;
config.total_resources = 100;

% 攻击类型配置
config.attack_types = {'DDoS', 'Malware', 'APT', 'Phishing', 'Zero-day', 'Insider'};
config.attack_severity = [0.3, 0.5, 0.8, 0.4, 0.9, 0.7];
config.attack_detection_difficulty = [0.2, 0.4, 0.8, 0.3, 0.9, 0.6];

% 资源类型配置
config.resource_types = {'Firewall', 'IDS', 'Antivirus', 'Encryption', 'Monitoring'};
config.resource_effectiveness = [0.7, 0.8, 0.6, 0.9, 0.75];

% FSP参数
config.fsp_alpha = 0.1;        % EWMA遗忘因子
config.w_radi = 0.5;           % RADI权重
config.w_damage = 0.5;         % 损害权重

% 攻击者Q-learning参数
config.attacker_lr = 0.1;
config.attacker_gamma = 0.95;
config.attacker_epsilon = 0.3;
config.attacker_epsilon_decay = 0.995;
config.attacker_epsilon_min = 0.01;

% 防御者RL参数
config.defender_lr = 0.05;
config.defender_gamma = 0.95;

% 仿真参数
n_episodes = 1000;
n_steps_per_episode = 100;

%% 2. 初始化环境
fprintf('初始化网络攻防博弈环境...\n');
env = TCSEnvironment(config);

%% 3. 初始化记录
episode_rewards = struct();
episode_rewards.attacker = zeros(n_episodes, 1);
episode_rewards.defender = zeros(n_episodes, 1);
episode_radi = zeros(n_episodes, 1);
episode_damage = zeros(n_episodes, 1);
episode_success_rate = zeros(n_episodes, 1);

% 策略历史
strategy_history = struct();
strategy_history.attacker = zeros(n_episodes, config.n_stations);
strategy_history.defender = zeros(n_episodes, config.n_stations);
strategy_history.perceived = zeros(n_episodes, config.n_stations);

%% 4. 运行博弈仿真
fprintf('开始FSP博弈仿真...\n');
fprintf('总回合数: %d × %d = %d\n', n_episodes, n_steps_per_episode, ...
        n_episodes * n_steps_per_episode);

for episode = 1:n_episodes
    % 重置环境
    state = env.reset();
    
    % 记录每轮的累积值
    episode_reward_att = 0;
    episode_reward_def = 0;
    episode_radi_sum = 0;
    episode_damage_sum = 0;
    episode_success_count = 0;
    
    for step = 1:n_steps_per_episode
        % 防御者决策：基于攻击者平均策略计算最佳响应
        defender_deployment = env.computeDefenderBestResponse();
        
        % 攻击者决策：基于防御部署选择攻击目标
        attacker_target = env.selectAttackerAction(defender_deployment);
        
        % 执行博弈步骤
        [next_state, reward_def, reward_att, info] = env.step(defender_deployment, attacker_target);
        
        % 累积奖励和指标
        episode_reward_att = episode_reward_att + reward_att;
        episode_reward_def = episode_reward_def + reward_def;
        episode_radi_sum = episode_radi_sum + info.radi_score;
        episode_damage_sum = episode_damage_sum + info.damage;
        if info.attack_success
            episode_success_count = episode_success_count + 1;
        end
        
        % 更新状态
        state = next_state;
    end
    
    % 记录回合统计
    episode_rewards.attacker(episode) = episode_reward_att;
    episode_rewards.defender(episode) = episode_reward_def;
    episode_radi(episode) = episode_radi_sum / n_steps_per_episode;
    episode_damage(episode) = episode_damage_sum / n_steps_per_episode;
    episode_success_rate(episode) = episode_success_count / n_steps_per_episode;
    
    % 记录策略
    strategy_history.attacker(episode, :) = env.attacker_strategy;
    strategy_history.defender(episode, :) = env.defender_strategy;
    strategy_history.perceived(episode, :) = env.attacker_avg_strategy;
    
    % 显示进度
    if mod(episode, 50) == 0
        fprintf('[Episode %d] RADI: %.3f, Damage: %.3f, Success Rate: %.2f%%\n', ...
                episode, episode_radi(episode), episode_damage(episode), ...
                episode_success_rate(episode) * 100);
        fprintf('  攻击策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.attacker_strategy);
        fprintf('  防守策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.defender_strategy);
        fprintf('  感知策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.attacker_avg_strategy);
    end
end

%% 5. 计算最终平均策略
% 防御者的最终策略：所有部署的平均
final_defense_strategy = mean(env.deployment_history, 1);
final_defense_strategy = final_defense_strategy / sum(final_defense_strategy);

% 攻击者的平均策略
attack_frequency = zeros(1, config.n_stations);
for i = 1:length(env.attack_history)
    attack_frequency(env.attack_history(i)) = attack_frequency(env.attack_history(i)) + 1;
end
final_attack_strategy = attack_frequency / sum(attack_frequency);

%% 6. 分析结果
fprintf('\n========== 博弈结果分析 ==========\n');
fprintf('平均RADI: %.4f\n', mean(episode_radi));
fprintf('平均损害: %.4f\n', mean(episode_damage));
fprintf('平均攻击成功率: %.2f%%\n', mean(episode_success_rate) * 100);
fprintf('\n最终策略:\n');
fprintf('实际攻击策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', final_attack_strategy);
fprintf('感知攻击策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.attacker_avg_strategy);
fprintf('防御部署策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', final_defense_strategy);

% 策略相似度
strategy_similarity = 1 - norm(final_attack_strategy - env.attacker_avg_strategy);
fprintf('\n策略相似度: %.2f%%\n', strategy_similarity * 100);

% 收敛性分析
last_100_radi = episode_radi(max(1, end-99):end);
convergence_score = 1 - std(last_100_radi) / mean(last_100_radi);
fprintf('收敛性得分: %.2f%%\n', convergence_score * 100);

%% 7. 可视化结果
figure('Name', 'FSP博弈分析', 'Position', [100, 100, 1400, 900]);

% 子图1: RADI演化
subplot(2,3,1);
window_size = 20;
radi_smooth = movmean(episode_radi, window_size);
plot(1:n_episodes, radi_smooth, 'b-', 'LineWidth', 2);
hold on;
plot([1, n_episodes], [0.15, 0.15], 'r--', 'LineWidth', 1.5);
xlabel('Episode');
ylabel('RADI');
title('RADI演化（目标: 0.15）');
grid on;
legend('RADI', '目标值', 'Location', 'best');

% 子图2: 攻击成功率
subplot(2,3,2);
success_smooth = movmean(episode_success_rate, window_size);
plot(1:n_episodes, success_smooth * 100, 'r-', 'LineWidth', 2);
xlabel('Episode');
ylabel('成功率 (%)');
title('攻击成功率演化');
grid on;
ylim([0, 100]);

% 子图3: 累积奖励
subplot(2,3,3);
plot(1:n_episodes, cumsum(episode_rewards.attacker), 'r-', 'LineWidth', 2);
hold on;
plot(1:n_episodes, cumsum(episode_rewards.defender), 'b-', 'LineWidth', 2);
xlabel('Episode');
ylabel('累积奖励');
title('累积奖励对比');
legend('攻击者', '防御者', 'Location', 'northwest');
grid on;

% 子图4: 策略演化热图（攻击者）
subplot(2,3,4);
sample_interval = max(1, floor(n_episodes / 50));
sampled_episodes = 1:sample_interval:n_episodes;
imagesc(1:config.n_stations, sampled_episodes, ...
        strategy_history.perceived(sampled_episodes, :));
colorbar;
xlabel('站点');
ylabel('Episode');
title('感知的攻击策略演化');

% 子图5: 最终策略对比
subplot(2,3,5);
strategies = [final_attack_strategy; 
              env.attacker_avg_strategy;
              final_defense_strategy];
bar(strategies');
xlabel('站点');
ylabel('概率/资源比例');
title('最终策略对比');
legend('实际攻击', '感知攻击', '防御部署', 'Location', 'best');
grid on;
set(gca, 'XTick', 1:config.n_stations);

% 子图6: 站点价值 vs 攻击频率
subplot(2,3,6);
yyaxis left;
bar(1:config.n_stations, env.station_values, 'FaceColor', [0.8, 0.8, 0.8]);
ylabel('站点价值');
ylim([0, max(env.station_values) * 1.2]);

yyaxis right;
plot(1:config.n_stations, final_attack_strategy, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('攻击频率');
ylim([0, max(final_attack_strategy) * 1.2]);

xlabel('站点');
title('站点价值 vs 攻击频率');
grid on;
set(gca, 'XTick', 1:config.n_stations);

sgtitle('基于FSP的网络攻防博弈分析');

%% 8. 保存结果
results = struct();
results.config = config;
results.episode_rewards = episode_rewards;
results.episode_radi = episode_radi;
results.episode_damage = episode_damage;
results.episode_success_rate = episode_success_rate;
results.strategy_history = strategy_history;
results.final_attack_strategy = final_attack_strategy;
results.final_defense_strategy = final_defense_strategy;
results.perceived_attack_strategy = env.attacker_avg_strategy;
results.station_values = env.station_values;
results.strategy_similarity = strategy_similarity;
results.convergence_score = convergence_score;

filename = sprintf('fsp_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
save(filename, 'results');
fprintf('\n结果已保存至: %s\n', filename);

%% 9. 详细策略分析（可选）
if true  % 设置为true以显示详细分析
    fprintf('\n========== 详细策略分析 ==========\n');
    
    % 分析每个站点
    for i = 1:config.n_stations
        fprintf('\n站点 %d:\n', i);
        fprintf('  价值: %.3f\n', env.station_values(i));
        fprintf('  被攻击频率: %.2f%%\n', final_attack_strategy(i) * 100);
        fprintf('  防御资源分配: %.2f%%\n', final_defense_strategy(i) * 100);
        fprintf('  感知威胁等级: %.2f%%\n', env.attacker_avg_strategy(i) * 100);
        
        % 计算防御效率
        if final_attack_strategy(i) > 0
            defense_efficiency = final_defense_strategy(i) / final_attack_strategy(i);
            fprintf('  防御效率: %.2f\n', defense_efficiency);
        end
    end
    
    % 分析RADI趋势
    fprintf('\n========== RADI趋势分析 ==========\n');
    recent_episodes = min(100, n_episodes);
    recent_radi = episode_radi(end-recent_episodes+1:end);
    fprintf('最近%d轮平均RADI: %.4f\n', recent_episodes, mean(recent_radi));
    fprintf('RADI标准差: %.4f\n', std(recent_radi));
    fprintf('RADI趋势: %s\n', analyzeTrend(recent_radi));
    
    % 学习效率分析
    fprintf('\n========== 学习效率分析 ==========\n');
    early_performance = mean(episode_success_rate(1:min(100, n_episodes)));
    late_performance = mean(episode_success_rate(max(1, end-99):end));
    learning_improvement = (early_performance - late_performance) / early_performance * 100;
    fprintf('早期攻击成功率: %.2f%%\n', early_performance * 100);
    fprintf('后期攻击成功率: %.2f%%\n', late_performance * 100);
    fprintf('防御改进率: %.2f%%\n', learning_improvement);
end

%% 辅助函数
function trend = analyzeTrend(data)
    % 分析数据趋势
    n = length(data);
    x = (1:n)';
    p = polyfit(x, data(:), 1);
    
    if p(1) < -0.0001
        trend = '下降';
    elseif p(1) > 0.0001
        trend = '上升';
    else
        trend = '稳定';
    end
end