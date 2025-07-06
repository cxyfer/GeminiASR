import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class ConfigManager:
    """
    配置管理器，負責載入和合併來自多個來源的配置
    優先級順序：命令列參數 > TOML 配置檔案 > 環境變數 > 預設值
    """
    
    def __init__(self):
        self.logger = logging.getLogger('gemini_asr.config')
        self._config = {}
        self._config_loaded = False
        
    def load_config(self) -> Dict[str, Any]:
        """
        載入配置，按照以下順序搜尋配置檔案：
        1. 當前目錄/config.toml
        2. 當前目錄/.geminiasr/config.toml
        3. ~/.geminiasr/config.toml
        4. ~/.config/geminiasr/config.toml
        
        Returns:
            Dict[str, Any]: 載入的配置字典
        """
        if self._config_loaded:
            return self._config
            
        config_paths = self._get_config_search_paths()
        
        for config_path in config_paths:
            if config_path.exists():
                self.logger.info(f"找到配置檔案: {config_path}")
                try:
                    self._config = self._load_toml_file(config_path)
                    self._config_loaded = True
                    return self._config
                except Exception as e:
                    self.logger.warning(f"載入配置檔案 {config_path} 失敗: {e}")
                    continue
        
        self.logger.info("未找到配置檔案，使用預設配置")
        self._config = {}
        self._config_loaded = True
        return self._config
    
    def _get_config_search_paths(self) -> List[Path]:
        """
        取得配置檔案搜尋路徑
        
        Returns:
            List[Path]: 配置檔案路徑列表
        """
        paths = []
        
        # 1. 當前目錄/config.toml
        paths.append(Path.cwd() / "config.toml")
        
        # 2. 當前目錄/.geminiasr/config.toml
        paths.append(Path.cwd() / ".geminiasr" / "config.toml")
        
        # 3. ~/.geminiasr/config.toml
        home = Path.home()
        paths.append(home / ".geminiasr" / "config.toml")
        
        # 4. ~/.config/geminiasr/config.toml
        config_home = home / ".config" / "geminiasr"
        paths.append(config_home / "config.toml")
        
        return paths
    
    def _load_toml_file(self, path: Path) -> Dict[str, Any]:
        """
        載入 TOML 檔案
        
        Args:
            path (Path): TOML 檔案路徑
            
        Returns:
            Dict[str, Any]: 解析後的配置字典
        """
        with open(path, 'rb') as f:
            return tomllib.load(f)
    
    def get_merged_config(self, args: Optional[Dict[str, Any]] = None, 
                         env_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        合併所有來源的配置，按照優先級順序
        
        Args:
            args (Optional[Dict[str, Any]]): 命令列參數
            env_vars (Optional[Dict[str, Any]]): 環境變數
            
        Returns:
            Dict[str, Any]: 合併後的配置
        """
        # 1. 載入 TOML 配置
        toml_config = self.load_config()
        
        # 2. 準備預設配置
        defaults = self._get_default_config()
        
        # 3. 合併配置（優先級：args > toml > env > defaults）
        merged = {}
        
        # 開始合併，從最低優先級開始
        merged.update(defaults)
        
        if env_vars:
            merged.update({k: v for k, v in env_vars.items() if v is not None})
        
        # 合併 TOML 配置
        if toml_config:
            merged.update(self._flatten_toml_config(toml_config))
        
        # 最後合併命令列參數（最高優先級）
        # 注意：這裡我們接受所有非 None 的值，包括 False
        if args:
            merged.update({k: v for k, v in args.items() if v is not None})
        
        return merged
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        取得預設配置
        
        Returns:
            Dict[str, Any]: 預設配置字典
        """
        return {
            'duration': 900,
            'lang': 'zh-TW',
            'model': 'gemini-2.5-flash-preview-05-20',
            'save_raw': False,
            'skip_existing': False,
            'preview': False,
            'max_segment_retries': 3,
            'ignore_keys_limit': False,
            'debug': False,
            'base_url': 'https://generativelanguage.googleapis.com/',
            'google_api_keys': [],
            'max_workers': None,
            'extra_prompt': None,
        }
    
    def _flatten_toml_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        將 TOML 巢狀配置扁平化為單層字典
        
        Args:
            config (Dict[str, Any]): TOML 配置字典
            
        Returns:
            Dict[str, Any]: 扁平化後的配置字典
        """
        flattened = {}
        
        # 轉錄相關設定
        if 'transcription' in config:
            t = config['transcription']
            flattened.update({
                'duration': t.get('duration'),
                'lang': t.get('lang'),
                'model': t.get('model'),
                'save_raw': t.get('save_raw'),
                'skip_existing': t.get('skip_existing'),
                'preview': t.get('preview'),
                'max_segment_retries': t.get('max_segment_retries'),
            })
        
        # 處理相關設定
        if 'processing' in config:
            p = config['processing']
            flattened.update({
                'max_workers': p.get('max_workers'),
                'ignore_keys_limit': p.get('ignore_keys_limit'),
            })
        
        # 日誌相關設定
        if 'logging' in config:
            l = config['logging']
            flattened.update({
                'debug': l.get('debug'),
            })
        
        # API 設定
        if 'api' in config:
            a = config['api']
            flattened.update({
                'google_api_keys': a.get('google_api_keys', []),
            })
        
        # 進階設定
        if 'advanced' in config:
            adv = config['advanced']
            flattened.update({
                'extra_prompt': adv.get('extra_prompt'),
                'base_url': adv.get('base_url'),
            })
        
        # 只移除 None 值，保留 False 和其他有效值
        return {k: v for k, v in flattened.items() if v is not None}
    
    def get_api_keys(self) -> List[str]:
        """
        取得 API 金鑰列表，依照優先級合併來源
        
        Returns:
            List[str]: API 金鑰列表
        """
        # 1. 從環境變數取得（包括 .env 文件）
        env_keys = []
        
        # 嘗試載入 .env 文件
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        env_keys_str = os.getenv('GOOGLE_API_KEY', '')
        if env_keys_str:
            env_keys = [key.strip() for key in env_keys_str.split(',') if key.strip()]
        
        # 2. 從 TOML 配置取得
        config = self.load_config()
        toml_keys = []
        if 'api' in config and 'google_api_keys' in config['api']:
            toml_keys = config['api']['google_api_keys']
        
        # 3. 優先使用 TOML 配置，如果沒有則使用環境變數
        if toml_keys:
            self.logger.info(f"使用 TOML 配置中的 {len(toml_keys)} 個 API 金鑰")
            return toml_keys
        elif env_keys:
            self.logger.info(f"使用環境變數中的 {len(env_keys)} 個 API 金鑰")
            return env_keys
        else:
            self.logger.warning("未找到任何 API 金鑰配置")
            return []
    
    def get_base_url(self) -> str:
        """
        取得 API 基礎 URL
        
        Returns:
            str: API 基礎 URL
        """
        # 嘗試載入 .env 文件
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        # 1. 檢查環境變數（優先級高於 TOML）
        env_base_url = os.getenv('BASE_URL')
        if env_base_url:
            return env_base_url
        
        # 2. 檢查 TOML 配置
        config = self.load_config()
        if 'advanced' in config and 'base_url' in config['advanced']:
            return config['advanced']['base_url']
        
        # 3. 使用預設值
        return 'https://generativelanguage.googleapis.com/'
    
    def log_config_details(self, config: Dict[str, Any], logger):
        """
        以 DEBUG 級別記錄所有配置設定的詳細資訊
        
        Args:
            config (Dict[str, Any]): 合併後的配置字典
            logger: 日誌記錄器實例
        """
        logger.debug("=== 配置設定詳細資訊 ===")
        logger.debug(f"輸入路徑: {config.get('input', 'N/A')}")
        logger.debug(f"分段時長: {config.get('duration', 'N/A')} 秒")
        logger.debug(f"語言設定: {config.get('lang', 'N/A')}")
        logger.debug(f"Gemini 模型: {config.get('model', 'N/A')}")
        logger.debug(f"保存原始轉錄: {config.get('save_raw', 'N/A')}")
        logger.debug(f"跳過已存在檔案: {config.get('skip_existing', 'N/A')}")
        logger.debug(f"預覽模式: {config.get('preview', 'N/A')}")
        logger.debug(f"偵錯模式: {config.get('debug', 'N/A')}")
        logger.debug(f"忽略金鑰限制: {config.get('ignore_keys_limit', 'N/A')}")
        logger.debug(f"分段最大重試次數: {config.get('max_segment_retries', 'N/A')}")
        logger.debug(f"API 基礎 URL: {config.get('base_url', 'N/A')}")
        logger.debug(f"載入的 API 金鑰數量: {len(config.get('google_api_keys', []))}")
        logger.debug(f"開始時間: {config.get('start', 'N/A')}")
        logger.debug(f"結束時間: {config.get('end', 'N/A')}")
        logger.debug(f"額外提示詞檔案: {config.get('extra_prompt', 'N/A')}")
        logger.debug("========================")


# 全域配置管理器實例
config_manager = ConfigManager()