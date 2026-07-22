"""
Application settings and configuration
"""

from typing import List, Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None


class Settings(BaseSettings):
    """Application settings"""
    
    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="allow"
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "allow"
    
    # Application
    app_name: str = "SS AI Server"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    
    # Database
    database_type: str = "postgresql"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "ss_ai_server"
    database_user: str = "postgres"
    database_password: str = "postgres"
    database_pool_size: int = 20
    database_max_overflow: int = 10
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_prefix: str = "ss_ai_cache"
    
    # Vector Store
    vector_store_type: str = "faiss"
    vector_store_dimensions: int = 512
    vector_store_metric: str = "cosine"
    vector_store_path: str = "/data/vector_store"
    
    # AI Provider
    ai_default_provider: str = "openclip"
    ai_default_model: str = "ViT-B-32"
    ai_device: str = "auto"
    ai_precision: str = "fp32"
    ai_cache_enabled: bool = True
    ai_cache_dir: str = "/models"
    
    # Security
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_default: int = 60
    rate_limit_window: int = 60
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]
    
    # Monitoring
    monitoring_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_outputs: List[str] = ["console", "file"]
    log_file_path: str = "/logs/ss_ai_server.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 10
    
    # Storage
    storage_type: str = "local"
    storage_path: str = "/data/storage"
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    # Queue
    queue_type: str = "redis"
    queue_redis_host: str = "localhost"
    queue_redis_port: int = 6379
    queue_redis_db: int = 1
    queue_max_retries: int = 3
    queue_retry_backoff: float = 2.0
    queue_job_timeout: int = 3600
    
    # Workers
    worker_embedding_enabled: bool = True
    worker_embedding_concurrency: int = 2
    worker_batch_enabled: bool = True
    worker_batch_concurrency: int = 1
    worker_cleanup_enabled: bool = True
    worker_cleanup_schedule: str = "0 3 * * *"
    
    # Analytics
    analytics_enabled: bool = True
    analytics_storage_type: str = "influxdb"
    analytics_retention: str = "90d"
    
    # Encryption
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    
    @property
    def database_url(self) -> str:
        """Get database URL"""
        if self.database_type == "postgresql":
            return f"postgresql+asyncpg://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
        elif self.database_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.database_name}.db"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"

settings = Settings()