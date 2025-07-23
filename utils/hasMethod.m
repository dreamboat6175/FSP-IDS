function result = hasMethod(obj, method_name)
    %% hasMethod - 检查对象是否有指定方法
    % 输入:
    %   obj - 要检查的对象
    %   method_name - 方法名称字符串
    % 输出:
    %   result - 布尔值，true表示有该方法
    
    try
        if isobject(obj)
            % 对于类对象，使用methods函数
            method_list = methods(obj);
            result = any(strcmp(method_list, method_name));
        elseif isstruct(obj)
            % 对于结构体，检查字段
            result = isfield(obj, method_name) && isa(obj.(method_name), 'function_handle');
        else
            result = false;
        end
    catch
        result = false;
    end
end
