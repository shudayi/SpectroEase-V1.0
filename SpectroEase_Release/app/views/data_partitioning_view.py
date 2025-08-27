class DataPartitioningView(QWidget):
    def __init__(self, plugins: dict):
        super().__init__()
        self.plugins = plugins
        self.method_list = QListWidget()
  

    def add_custom_algorithm(self, code):
        """添加自定义data分割算法"""
        try:
            import types
            mod = types.ModuleType('custom_splitter')
            exec(code, mod.__dict__)
            
            for item in mod.__dict__.values():
                if isinstance(item, type):
                    algorithm = item()
                    self.plugins[algorithm.name] = algorithm
                    self.method_list.addItem(algorithm.name)
                    break
                    
        except Exception as e:
            raise Exception(f"Error loading custom splitter: {str(e)}") 