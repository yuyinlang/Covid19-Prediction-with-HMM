import inspect


def get_all_functions(module):
    method_list = [func for func in dir(HMM) if callable(getattr(HMM, func))]
    #functions = [i for i, _ in inspect.getmembers(module, inspect.isfunction)]
    return functions


def check_for_function(name, module):
    return True if name in get_all_functions(module) else False


def check_imported_libraries(module):
    libs = [i for i, _ in inspect.getmembers(module, inspect.ismodule)]
    libs_given = ['os', 'np', 'vis', 'ut', 'visualization', 'helpers', 'helpers.utils']
    flag = True
    for lib in libs:
        if lib not in libs_given:
            print(lib)
            flag = False
    return flag
