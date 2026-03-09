# OpenResty 反代 10500 gRPC 说明

本项目默认以单端口 gRPC 模式运行：`src/config.py` 中 `PORT` 与 `GRPC_PORT` 默认都是 `10500`，而 `src/main.py` 会在两者相同的时候自动关闭 HTTP/JSON，只保留 gRPC。

因此：

- 若直接反代默认的 `10500`，应使用 `grpc_pass`，而不是 `proxy_pass`。
- 若由 OpenResty 负责外部 TLS，建议让后端保持默认 h2c（明文 HTTP/2）监听 `127.0.0.1:10500`。
- 若后端服务也要启用 TLS，则把 `deploy/openresty/proxy/grpc_10500.conf` 中的 `grpc_pass grpc://...` 改成 `grpc_pass grpcs://...`，并补充 `grpc_ssl_*` 配置。

## 文件位置

- 站点配置：`deploy/openresty/ca.macrz.com.conf`
- gRPC 代理片段：`deploy/openresty/proxy/grpc_10500.conf`

## 部署建议

1. 把 `deploy/openresty/ca.macrz.com.conf` 放到 OpenResty 站点配置目录。
2. 把 `deploy/openresty/proxy/grpc_10500.conf` 放到 `/www/sites/ca.macrz.com/proxy/`。
3. 确保后端服务监听宿主机 `127.0.0.1:10500` 或对应可达地址。
4. 若需要双向 HTTPS（mTLS），在站点配置里启用 `ssl_client_certificate` 与 `ssl_verify_client on`。

## 验证命令

- 语法检查：`openresty -t` 或 `nginx -t`
- gRPC 健康检查：使用 TLS 客户端调用 `grpc.health.v1.Health/Check`
- 流式校验：调用 `com.continuousauth.proto.SensorDataService/StreamSensorData`

本次仓库内联调按上面这套方式完成：OpenResty 对外使用 `443/TLS+h2`，上游回源到本项目默认的 `127.0.0.1:10500(h2c)`。
