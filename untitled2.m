%% analyze_class_structure.m - 深度分析MATLAB类结构
% =========================================================================
% 描述: 详细分析类文件结构，找出实际的重复定义问题
% =========================================================================

function analyze_class_structure()
    fprintf('=== 深度分析类文件结构 ===\n\n');
    
    % 分析主要的类文件
    class_files = {
        'agents/DoubleQLearningAgent.m',
        'agents/QLearningAgent.m',
        'agents/SARSAAgent.m',
        'core/PerformanceMonitor.m'
    };
    
    for i = 1:length(class_files)
        if exist(class_files{i}, 'file')
            fprintf('分析文件: %s\n', class_files{i});
            analyze_single_file(class_files{i});
            fprintf('\n');
        end
    end
end

function analyze_single_file(filename)
    % 读取文件
    content = fileread(filename);
    lines = strsplit(content, '\n');
    
    % 查找所有函数定义
    functions_found = {};
    function_lines = [];
    
    for i = 1:length(lines)
        line = lines{i};
        
        % 查找函数定义（包括构造函数）
        % 匹配各种函数定义格式
        patterns = {
            'function\s+(\w+)\s*\(',                    % function name(
            'function\s+\w+\s*=\s*(\w+)\s*\(',         % function out = name(
            'function\s+\[.*?\]\s*=\s*(\w+)\s*\(',     % function [out1,out2] = name(
        };
        
        for p = 1:length(patterns)
            matches = regexp(line, patterns{p}, 'tokens');
            if ~isempty(matches)
                func_name = matches{1}{1};
                functions_found{end+1} = func_name;
                function_lines(end+1) = i;
                break;
            end
        end
    end
    
    % 查找重复
    unique_funcs = unique(functions_found);
    fprintf('  总函数数: %d\n', length(functions_found));
    fprintf('  唯一函数数: %d\n', length(unique_funcs));
    
    % 显示重复函数
    for i = 1:length(unique_funcs)
        func_name = unique_funcs{i};
        indices = find(strcmp(functions_found, func_name));
        if length(indices) > 1
            fprintf('  🔄 函数 "%s" 出现 %d 次:\n', func_name, length(indices));
            for j = 1:length(indices)
                line_num = function_lines(indices(j));
                fprintf('     - 第 %d 行: %s\n', line_num, strtrim(lines{line_num}));
            end
        end
    end
    
    % 检查特殊情况：类定义中的方法
    check_class_structure(lines);
end

function check_class_structure(lines)
    % 检查类的结构问题
    fprintf('  \n  类结构分析:\n');
    
    % 查找classdef
    classdef_line = 0;
    for i = 1:length(lines)
        if contains(lines{i}, 'classdef')
            classdef_line = i;
            fprintf('    - classdef 在第 %d 行\n', i);
            break;
        end
    end
    
    % 查找所有methods块
    methods_blocks = [];
    methods_count = 0;
    in_methods = false;
    
    for i = 1:length(lines)
        line = strtrim(lines{i});
        
        % 检测methods块开始
        if startsWith(line, 'methods')
            methods_count = methods_count + 1;
            methods_blocks(end+1) = i;
            in_methods = true;
            fprintf('    - methods块 #%d 在第 %d 行\n', methods_count, i);
            
            % 检查methods的属性
            if contains(line, 'Static')
                fprintf('      (Static methods)\n');
            elseif contains(line, 'Access')
                fprintf('      (带访问控制)\n');
            end
        end
        
        % 检测对应的end
        if in_methods && strcmp(line, 'end')
            in_methods = false;
        end
    end
    
    % 分析问题
    if methods_count > 3
        fprintf('  ⚠️ 警告: 发现 %d 个methods块，可能存在结构问题\n', methods_count);
    end
end

% 新增：修复类文件结构问题
function fix_class_structure_issues()
    fprintf('\n=== 修复类文件结构问题 ===\n');
    
    % 需要修复的文件
    files_to_fix = {
        'agents/QLearningAgent.m',
        'agents/SARSAAgent.m',
        'agents/DoubleQLearningAgent.m'
    };
    
    for i = 1:length(files_to_fix)
        if exist(files_to_fix{i}, 'file')
            fprintf('\n修复文件: %s\n', files_to_fix{i});
            fix_single_class_file(files_to_fix{i});
        end
    end
end

function fix_single_class_file(filename)
    % 读取文件
    content = fileread(filename);
    lines = strsplit(content, '\n');
    
    % 收集所有方法定义
    methods_info = collect_all_methods(lines);
    
    % 重构类文件
    new_lines = restructure_class_file(lines, methods_info);
    
    % 备份并保存
    backup_file(filename);
    
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', strjoin(new_lines, '\n'));
    fclose(fid);
    
    fprintf('✓ 文件已重构\n');
end

function methods_info = collect_all_methods(lines)
    methods_info = struct();
    methods_info.constructor = [];
    methods_info.public_methods = [];
    methods_info.private_methods = [];
    methods_info.static_methods = [];
    
    current_method = [];
    in_method = false;
    method_depth = 0;
    
    for i = 1:length(lines)
        line = lines{i};
        
        % 检测函数定义
        func_match = regexp(line, 'function\s+(?:\[?.*?\]?\s*=\s*)?(\w+)\s*\(', 'tokens');
        if ~isempty(func_match)
            if ~isempty(current_method)
                % 保存前一个方法
                methods_info = add_method_to_info(methods_info, current_method);
            end
            
            % 开始新方法
            current_method = struct();
            current_method.name = func_match{1}{1};
            current_method.start_line = i;
            current_method.lines = {line};
            current_method.type = determine_method_type(current_method.name, line);
            in_method = true;
            method_depth = count_indent(line);
            continue;
        end
        
        % 收集方法内容
        if in_method
            current_method.lines{end+1} = line;
            
            % 检查方法结束
            if contains(strtrim(line), 'end') && count_indent(line) == method_depth
                in_method = false;
                methods_info = add_method_to_info(methods_info, current_method);
                current_method = [];
            end
        end
    end
    
    % 保存最后一个方法
    if ~isempty(current_method)
        methods_info = add_method_to_info(methods_info, current_method);
    end
end

function methods_info = add_method_to_info(methods_info, method)
    % 根据方法类型添加到相应的列表
    switch method.type
        case 'constructor'
            % 只保留第一个构造函数
            if isempty(methods_info.constructor)
                methods_info.constructor = method;
            else
                fprintf('  ⚠️ 发现重复构造函数 %s，已忽略\n', method.name);
            end
        case 'static'
            methods_info.static_methods{end+1} = method;
        case 'private'
            methods_info.private_methods{end+1} = method;
        otherwise
            methods_info.public_methods{end+1} = method;
    end
end

function type = determine_method_type(name, line)
    % 判断方法类型
    if endsWith(name, 'Agent')  % 构造函数
        type = 'constructor';
    elseif contains(line, 'obj') || contains(line, 'self')
        type = 'public';
    else
        type = 'static';
    end
end

function new_lines = restructure_class_file(lines, methods_info)
    % 重新构建类文件
    new_lines = {};
    
    % 1. 复制classdef行
    for i = 1:length(lines)
        if contains(lines{i}, 'classdef')
            new_lines{end+1} = lines{i};
            break;
        end
    end
    
    % 2. 复制properties块
    in_props = false;
    for i = 1:length(lines)
        if contains(lines{i}, 'properties')
            in_props = true;
        end
        if in_props
            new_lines{end+1} = lines{i};
            if contains(strtrim(lines{i}), 'end') && ~contains(lines{i}, 'end;')
                in_props = false;
            end
        end
    end
    
    % 3. 添加methods块（公共方法）
    new_lines{end+1} = '    ';
    new_lines{end+1} = '    methods';
    
    % 添加构造函数
    if ~isempty(methods_info.constructor)
        for j = 1:length(methods_info.constructor.lines)
            new_lines{end+1} = methods_info.constructor.lines{j};
        end
        new_lines{end+1} = '        ';
    end
    
    % 添加公共方法（去重）
    added_methods = {};
    for i = 1:length(methods_info.public_methods)
        method = methods_info.public_methods{i};
        if ~ismember(method.name, added_methods)
            for j = 1:length(method.lines)
                new_lines{end+1} = method.lines{j};
            end
            new_lines{end+1} = '        ';
            added_methods{end+1} = method.name;
        else
            fprintf('  ✓ 删除重复方法: %s\n', method.name);
        end
    end
    
    new_lines{end+1} = '    end';
    
    % 4. 添加静态方法块（如果有）
    if ~isempty(methods_info.static_methods)
        new_lines{end+1} = '    ';
        new_lines{end+1} = '    methods (Static)';
        
        added_methods = {};
        for i = 1:length(methods_info.static_methods)
            method = methods_info.static_methods{i};
            if ~ismember(method.name, added_methods)
                for j = 1:length(method.lines)
                    new_lines{end+1} = method.lines{j};
                end
                new_lines{end+1} = '        ';
                added_methods{end+1} = method.name;
            end
        end
        
        new_lines{end+1} = '    end';
    end
    
    % 5. 结束类定义
    new_lines{end+1} = 'end';
end

function indent = count_indent(line)
    % 计算缩进级别
    indent = 0;
    for i = 1:length(line)
        if line(i) == ' '
            indent = indent + 1;
        else
            break;
        end
    end
end

function backup_file(filename)
    [path, name, ext] = fileparts(filename);
    backup_name = fullfile(path, [name '_backup_' datestr(now, 'yyyymmdd_HHMMSS') ext]);
    copyfile(filename, backup_name);
end

%% 运行分析
if ~isdeployed
    analyze_class_structure();
    
    % 询问是否要修复
    answer = input('\n是否要自动修复发现的问题？(y/n): ', 's');
    if strcmpi(answer, 'y')
        fix_class_structure_issues();
    end
end