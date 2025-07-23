function result = hasProperty(obj, prop_name)
    %% hasProperty - 检查对象是否有指定属性
    % 输入:
    %   obj - 要检查的对象
    %   prop_name - 属性名称字符串
    % 输出:
    %   result - 布尔值，true表示有该属性
    
    try
        if isobject(obj)
            % 对于类对象，使用isprop
            result = isprop(obj, prop_name);
        elseif isstruct(obj)
            % 对于结构体，使用isfield
            result = isfield(obj, prop_name);
        else
            result = false;
        end
    catch
        result = false;
    end
end