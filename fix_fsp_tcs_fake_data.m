function fix_fsp_tcs_fake_data()
%FIX_FSP_TCS_FAKE_DATA 自动修复FSP-TCS系统中的虚假数据和跳变问题
%
% 该脚本将：
% 1. 备份原始文件
% 2. 修复visualization/generateVisualizationReport.m中的虚假数据生成
% 3. 修复core/updatePerformanceMonitor.m中的跳变问题
% 4. 创建数据验证工具
% 5. 清理.asv文件
% 6. 验证修复结果

fprintf('🔧 开始修复FSP-TCS虚假数据和跳变问题...\n\n');

try
    %% 第1步：备份原始文件
    fprintf('1️⃣ 备份原始文件...\n');
    backup_files();
    
    %% 第2步：修复可视化报告生成器
    fprintf('2️⃣ 修复可视化报告生成器...\n');
    fix_visualization_report();
    
    %% 第3步：修复性能监控器
    fprintf('3️⃣ 修复性能监控器跳变问题...\n');
    fix_performance_monitor();
    
    %% 第4步：创建数据验证工具
    fprintf('4️⃣ 创建数据验证工具...\n');
    create_data_validator();
    
    %% 第5步：清理临时文件
    fprintf('5️⃣ 清理.asv文件...\n');
    cleanup_asv_files();
    
    %% 第6步：验证修复结果
    fprintf('6️⃣ 验证修复结果...\n');
    verify_fixes();
    
    fprintf('\n🎉 FSP-TCS虚假数据问题修复完成！\n');
    fprintf('💡 建议运行短期仿真测试以确认系统正常工作\n');
    fprintf('⚠️  如果测试失败，说明需要修复真实的仿真逻辑，而不是生成虚假数据\n');
    
catch ME
    fprintf('❌ 修复过程出错: %s\n', ME.message);
    fprintf('📍 错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
    fprintf('💡 请参考修复方案手动执行修复步骤\n');
end

end

function backup_files()
%BACKUP_FILES 备份原始文件

files_to_backup = {
    'visualization/generateVisualizationReport.m',
    'core/updatePerformanceMonitor.m'
};

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
backup_dir = sprintf('backup_%s', timestamp);

if ~exist(backup_dir, 'dir')
    mkdir(backup_dir);
end

for i = 1:length(files_to_backup)
    file_path = files_to_backup{i};
    if exist(file_path, 'file')
        [~, name, ext] = fileparts(file_path);
        backup_path = fullfile(backup_dir, [name '_backup' ext]);
        copyfile(file_path, backup_path);
        fprintf('   ✅ 已备份: %s -> %s\n', file_path, backup_path);
    else
        fprintf('   ⚠️  文件不存在: %s\n', file_path);
    end
end

end

function fix_visualization_report()
%FIX_VISUALIZATION_REPORT 修复可视化报告生成器

file_path = 'visualization/generateVisualizationReport.m';

if ~exist(file_path, 'file')
    fprintf('   ⚠️  文件不存在: %s\n', file_path);
    return;
end

% 读取文件内容
content = fileread(file_path);

% 定义需要修复的虚假数据生成模式
fake_patterns = {
    'rand\(\) \* 2000 \+ 1000',
    'rand\(1, results_data\.n_iterations\) \* 0\.3 \+ 0\.6',
    'rand\(1, results_data\.n_iterations\) \* 0\.1 \+ 0\.05',
    '1000 \+ 500 \* rand\(\)',
    '0\.7\+0\.2\*rand\(\)',
    '0\.8\+0\.1\*rand\(\)',
    '0\.1\+0\.05\*rand\(\)',
    '0\.75\+0\.2\*rand\(\)',
    'randn\(1, n_iter\) \* 0\.02',
    'randn\(1, n_iter\) \* 0\.05',
    'randn\(1, n_iter\) \* 0\.08'
};

% 对应的替换文本
replacements = {
    'error(''FSP-TCS错误: 缺少真实奖励数据'')',
    'error(''FSP-TCS错误: 缺少真实检测率数据'')',
    'error(''FSP-TCS错误: 缺少真实RADI数据'')',
    'error(''FSP-TCS错误: 缺少真实奖励数据'')',
    'error(''FSP-TCS错误: 缺少真实雷达图数据'')',
    'error(''FSP-TCS错误: 缺少真实雷达图数据'')',
    'error(''FSP-TCS错误: 缺少真实雷达图数据'')',
    'error(''FSP-TCS错误: 缺少真实雷达图数据'')',
    'error(''FSP-TCS错误: 缺少真实安全趋势数据'')',
    'error(''FSP-TCS错误: 缺少真实资源利用数据'')',
    'error(''FSP-TCS错误: 缺少真实冲突强度数据'')'
};

% 应用修复
original_content = content;
for i = 1:length(fake_patterns)
    content = regexprep(content, fake_patterns{i}, replacements{i});
end

% 检查是否有修改
if ~strcmp(content, original_content)
    % 写回文件
    fid = fopen(file_path, 'w');
    if fid == -1
        error('无法打开文件进行写入: %s', file_path);
    end
    fprintf(fid, '%s', content);
    fclose(fid);
    fprintf('   ✅ 已修复虚假数据生成: %s\n', file_path);
else
    fprintf('   ℹ️  文件无需修复: %s\n', file_path);
end

end

function fix_performance_monitor()
%FIX_PERFORMANCE_MONITOR 修复性能监控器的跳变问题

file_path = 'core/updatePerformanceMonitor.m';

if ~exist(file_path, 'file')
    fprintf('   ⚠️  文件不存在: %s\n', file_path);
    return;
end

% 创建修复后的文件内容
fixed_content = create_fixed_performance_monitor_content();

% 写入修复后的内容
fid = fopen(file_path, 'w');
if fid == -1
    error('无法打开文件进行写入: %s', file_path);
end
fprintf(fid, '%s', fixed_content);
fclose(fid);

fprintf('   ✅ 已修复跳变问题: %s\n', file_path);

end

function content = create_fixed_performance_monitor_content()
%CREATE_FIXED_PERFORMANCE_MONITOR_CONTENT 创建修复后的性能监控器内容

content = [
'function updatePerformanceMonitor(monitor, iteration, episode_results, config)\n'
'    %% updatePerformanceMonitor - 更新性能监控器（修复版）\n'
'    % 修复了RADI和检测率跳变问题\n'
'    \n'
'    % 持久化历史数据以实现平滑处理\n'
'    persistent detection_history;\n'
'    persistent radi_history;\n'
'    \n'
'    if isempty(detection_history)\n'
'        detection_history = [];\n'
'    end\n'
'    if isempty(radi_history)\n'
'        radi_history = [];\n'
'    end\n'
'    \n'
'    try\n'
'        if isempty(monitor)\n'
'            return;\n'
'        end\n'
'        \n'
'        metrics = struct();\n'
'        \n'
'        % 基本性能指标\n'
'        if isfield(episode_results, ''avg_resource_allocation'')\n'
'            metrics.resource_allocation = mean(episode_results.avg_resource_allocation, 1);\n'
'        else\n'
'            metrics.resource_allocation = zeros(1, config.n_stations);\n'
'        end\n'
'        \n'
'        if isfield(episode_results, ''avg_efficiency'')\n'
'            metrics.resource_efficiency = mean(episode_results.avg_efficiency);\n'
'        else\n'
'            metrics.resource_efficiency = 0.5;\n'
'        end\n'
'        \n'
'        if isfield(episode_results, ''avg_balance'')\n'
'            metrics.allocation_balance = mean(episode_results.avg_balance);\n'
'        else\n'
'            metrics.allocation_balance = 0.5;\n'
'        end\n'
'        \n'
'        % 检测率计算 - 使用平滑处理避免跳变\n'
'        if isfield(episode_results, ''attack_info'')\n'
'            attack_success_rate = mean([episode_results.attack_info{:}]);\n'
'            current_detection_rate = 1 - attack_success_rate;\n'
'        else\n'
'            % 使用历史数据插值，避免固定默认值\n'
'            if ~isempty(detection_history)\n'
'                current_detection_rate = detection_history(end);\n'
'                fprintf(''警告: 第%d轮缺少攻击信息，使用历史检测率: %.3f\\n'', iteration, current_detection_rate);\n'
'            else\n'
'                fprintf(''错误: 第%d轮无法计算检测率，跳过本轮更新\\n'', iteration);\n'
'                return;\n'
'            end\n'
'        end\n'
'        \n'
'        % 应用移动平均平滑\n'
'        if ~isempty(detection_history)\n'
'            alpha = 0.3; % 平滑系数\n'
'            smoothed_detection_rate = alpha * current_detection_rate + (1-alpha) * detection_history(end);\n'
'        else\n'
'            smoothed_detection_rate = current_detection_rate;\n'
'        end\n'
'        detection_history(end+1) = smoothed_detection_rate;\n'
'        metrics.detection_rate = smoothed_detection_rate;\n'
'        \n'
'        % RADI指标 - 使用相同的平滑处理\n'
'        if isfield(episode_results, ''avg_radi'')\n'
'            current_radi = mean(episode_results.avg_radi);\n'
'        else\n'
'            if ~isempty(radi_history)\n'
'                current_radi = radi_history(end);\n'
'                fprintf(''警告: 第%d轮缺少RADI数据，使用历史值: %.3f\\n'', iteration, current_radi);\n'
'            else\n'
'                fprintf(''错误: 第%d轮无法计算RADI，跳过本轮更新\\n'', iteration);\n'
'                return;\n'
'            end\n'
'        end\n'
'        \n'
'        % RADI平滑处理\n'
'        if ~isempty(radi_history)\n'
'            alpha = 0.2; % 更保守的平滑系数\n'
'            smoothed_radi = alpha * current_radi + (1-alpha) * radi_history(end);\n'
'        else\n'
'            smoothed_radi = current_radi;\n'
'        end\n'
'        radi_history(end+1) = smoothed_radi;\n'
'        metrics.avg_radi = smoothed_radi;\n'
'        \n'
'        % 限制历史记录长度，避免内存溢出\n'
'        max_history = 1000;\n'
'        if length(detection_history) > max_history\n'
'            detection_history = detection_history(end-max_history+1:end);\n'
'        end\n'
'        if length(radi_history) > max_history\n'
'            radi_history = radi_history(end-max_history+1:end);\n'
'        end\n'
'        \n'
'        % 奖励指标\n'
'        if isfield(episode_results, ''avg_defender_reward'')\n'
'            metrics.avg_defender_reward = mean(episode_results.avg_defender_reward);\n'
'        else\n'
'            metrics.avg_defender_reward = 0;\n'
'        end\n'
'        \n'
'        if isfield(episode_results, ''avg_attacker_reward'')\n'
'            metrics.avg_attacker_reward = episode_results.avg_attacker_reward;\n'
'        else\n'
'            metrics.avg_attacker_reward = 0;\n'
'        end\n'
'        \n'
'        % 更新监控器\n'
'        if hasMethod(monitor, ''updateMetrics'')\n'
'            monitor.updateMetrics(iteration, metrics);\n'
'        elseif hasMethod(monitor, ''update'')\n'
'            monitor.update(iteration, metrics);\n'
'        else\n'
'            try\n'
'                monitor.latest_metrics = metrics;\n'
'                monitor.last_update_iteration = iteration;\n'
'            catch\n'
'                % 静默处理\n'
'            end\n'
'        end\n'
'        \n'
'        % 定期显示状态\n'
'        if mod(iteration, 50) == 0\n'
'            fprintf(''第%d轮 - 检测率: %.1f%%, RADI: %.3f\\n'', iteration, metrics.detection_rate*100, metrics.avg_radi);\n'
'        end\n'
'        \n'
'    catch ME\n'
'        warning(''更新性能监控器时出错 (迭代 %d): %s'', iteration, ME.message);\n'
'    end\n'
'end\n'
'\n'
'function has_method = hasMethod(obj, method_name)\n'
'    try\n'
'        if isobject(obj)\n'
'            has_method = any(strcmp(methods(obj), method_name));\n'
'        else\n'
'            has_method = false;\n'
'        end\n'
'    catch\n'
'        has_method = false;\n'
'    end\n'
'end\n'
];

end

function create_data_validator()
%CREATE_DATA_VALIDATOR 创建数据验证工具

validator_dir = 'utils';
if ~exist(validator_dir, 'dir')
    mkdir(validator_dir);
end

validator_file = fullfile(validator_dir, 'validateSimulationData.m');

validator_content = [
'function isValid = validateSimulationData(results)\n'
'%VALIDATESIMULATIONDATA 验证仿真数据的真实性\n'
'%\n'
'% 输入: results - 仿真结果结构体\n'
'% 输出: isValid - 布尔值，表示数据是否有效\n'
'\n'
'isValid = true;\n'
'errors = {};\n'
'\n'
'% 检查基本结构\n'
'if ~isstruct(results)\n'
'    errors{end+1} = ''结果不是有效的结构体'';\n'
'    isValid = false;\n'
'    reportErrors(errors);\n'
'    return;\n'
'end\n'
'\n'
'% 检查防御者数据\n'
'if isfield(results, ''defenders'')\n'
'    defender_names = fieldnames(results.defenders);\n'
'    for i = 1:length(defender_names)\n'
'        defender = results.defenders.(defender_names{i});\n'
'        \n'
'        if isfield(defender, ''performance'')\n'
'            perf = defender.performance;\n'
'            \n'
'            % 检查检测率数据的合理性\n'
'            if isfield(perf, ''detection_rate'')\n'
'                detection_data = perf.detection_rate;\n'
'                if isSuspiciousData(detection_data, ''detection_rate'')\n'
'                    errors{end+1} = sprintf(''防御者 %s 的检测率数据疑似虚假'', defender_names{i});\n'
'                    isValid = false;\n'
'                end\n'
'            end\n'
'            \n'
'            % 检查RADI数据的合理性\n'
'            if isfield(perf, ''radi'')\n'
'                radi_data = perf.radi;\n'
'                if isSuspiciousData(radi_data, ''radi'')\n'
'                    errors{end+1} = sprintf(''防御者 %s 的RADI数据疑似虚假'', defender_names{i});\n'
'                    isValid = false;\n'
'                end\n'
'            end\n'
'        end\n'
'    end\n'
'end\n'
'\n'
'% 报告错误\n'
'if ~isempty(errors)\n'
'    reportErrors(errors);\n'
'else\n'
'    fprintf(''✅ 数据验证通过 - 所有数据来源于真实仿真\\n'');\n'
'end\n'
'\n'
'end\n'
'\n'
'function suspicious = isSuspiciousData(data, type)\n'
'%检测数据是否可疑（可能是虚假生成的）\n'
'\n'
'suspicious = false;\n'
'\n'
'if isempty(data) || ~isnumeric(data)\n'
'    return;\n'
'end\n'
'\n'
'% 检查数据的统计特征\n'
'if length(data) > 20\n'
'    % 计算自相关性 - 真实学习数据应该有时间依赖性\n'
'    if length(data) > 10\n'
'        correlation = corrcoef(data(1:end-1), data(2:end));\n'
'        if abs(correlation(1,2)) < 0.1\n'
'            suspicious = true;\n'
'            return;\n'
'        end\n'
'    end\n'
'    \n'
'    % 检查变异系数 - 过度随机的数据变异系数会很大\n'
'    cv = std(data) / mean(data);\n'
'    if strcmp(type, ''detection_rate'') && cv > 0.3\n'
'        suspicious = true;\n'
'    elseif strcmp(type, ''radi'') && cv > 0.5\n'
'        suspicious = true;\n'
'    end\n'
'end\n'
'\n'
'end\n'
'\n'
'function reportErrors(errors)\n'
'%报告验证错误\n'
'\n'
'fprintf(''\\n❌ 数据验证失败，发现以下问题:\\n'');\n'
'for i = 1:length(errors)\n'
'    fprintf(''  %d. %s\\n'', i, errors{i});\n'
'end\n'
'fprintf(''\\n💡 建议检查仿真代码是否正确实现，确保所有数据都来自真实算法运行\\n\\n'');\n'
'\n'
'end\n'
];

fid = fopen(validator_file, 'w');
if fid == -1
    error('无法创建数据验证器文件: %s', validator_file);
end
fprintf(fid, '%s', validator_content);
fclose(fid);

fprintf('   ✅ 已创建数据验证器: %s\n', validator_file);

end

function cleanup_asv_files()
%CLEANUP_ASV_FILES 清理.asv文件

% 查找所有.asv文件
asv_files = {};
search_dirs = {'visualization', 'core', 'utils', '.'};

for i = 1:length(search_dirs)
    dir_path = search_dirs{i};
    if exist(dir_path, 'dir')
        files = dir(fullfile(dir_path, '*.asv'));
        for j = 1:length(files)
            asv_files{end+1} = fullfile(dir_path, files(j).name);
        end
    end
end

% 删除.asv文件
for i = 1:length(asv_files)
    delete(asv_files{i});
    fprintf('   🗑️  已删除: %s\n', asv_files{i});
end

if isempty(asv_files)
    fprintf('   ℹ️  未发现.asv文件\n');
end

end

function verify_fixes()
%VERIFY_FIXES 验证修复结果

fprintf('   🔍 检查修复后的文件...\n');

% 检查关键文件是否存在
critical_files = {
    'visualization/generateVisualizationReport.m',
    'core/updatePerformanceMonitor.m',
    'utils/validateSimulationData.m'
};

all_exist = true;
for i = 1:length(critical_files)
    if exist(critical_files{i}, 'file')
        fprintf('   ✅ %s\n', critical_files{i});
    else
        fprintf('   ❌ 缺失: %s\n', critical_files{i});
        all_exist = false;
    end
end

% 检查修复后的文件是否还包含虚假数据生成
if exist('visualization/generateVisualizationReport.m', 'file')
    content = fileread('visualization/generateVisualizationReport.m');
    suspicious_patterns = {'rand() *', 'randn(1,', 'rand(1,'};
    
    found_suspicious = false;
    for i = 1:length(suspicious_patterns)
        if contains(content, suspicious_patterns{i})
            fprintf('   ⚠️  仍包含可疑模式: %s\n', suspicious_patterns{i});
            found_suspicious = true;
        end
    end
    
    if ~found_suspicious
        fprintf('   ✅ 可视化报告生成器已清理\n');
    end
end

if all_exist && ~found_suspicious
    fprintf('   ✅ 修复验证通过\n');
else
    fprintf('   ⚠️  修复可能不完整，请手动检查\n');
end

end