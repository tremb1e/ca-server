# OpenResty 反代 10500 gRPC 说明

本项目默认以单端口 gRPC 模式运行：`src/config.py` 中 `PORT` 与 `GRPC_PORT` 默认都是 `10500`，而 `src/main.py` 会在两者相同的时候自动关闭 HTTP/JSON，只保留 gRPC。

因此：

- 若直接反代默认的 `10500`，应使用 `grpc_pass`，而不是 `proxy_pass`。
- 若由 OpenResty 负责外部 TLS，建议让后端保持默认 h2c（明文 HTTP/2）。若 OpenResty 与后端在同机，回源 `127.0.0.1:10500`；若 OpenResty 在远端机器，回源后端真实可达地址（例如内网 IP 或业务域名）的 `10500`。
- 若后端服务也要启用 TLS，则把 `deploy/openresty/proxy/grpc_10500.conf` 中的 `grpc_pass grpc://...` 改成 `grpc_pass grpcs://...`，并补充 `grpc_ssl_*` 配置。
- 容器化部署并不会把 gRPC 协议改成别的协议。当前推荐 compose 方案使用 `network_mode: host`，容器内 gRPC 直接监听宿主机 `10500`，以对齐 `python3 -m src.main` 的直跑网络形态。
- OpenResty / Nginx 默认 `client_max_body_size` 往往只有 `1m`；而本服务 gRPC 端默认允许 `4MiB` 单消息。若 App 单个 `DataPacket` 超过 1MiB，小于 4MiB，会出现“直连可用，反代失败”的典型现象。
- `SensorDataService/StreamSensorData` 是双向流式 RPC，代理层还需要足够长的 `grpc_read_timeout` / `grpc_send_timeout`，否则空闲一段时间后会被提前断流。

## 文件位置

- 站点配置：`deploy/openresty/ca.macrz.com.conf`
- gRPC 代理片段：`deploy/openresty/proxy/grpc_10500.conf`

## 部署建议

1. 把 `deploy/openresty/ca.macrz.com.conf` 放到 OpenResty 站点配置目录。
2. 把 `deploy/openresty/proxy/grpc_10500.conf` 放到 `/www/sites/ca.macrz.com/proxy/`。
3. 确保后端服务监听宿主机 `10500`，并让 OpenResty 按其部署位置回源：
   - 同机 OpenResty：`127.0.0.1:10500`
   - 远端 OpenResty：后端真实可达地址的 `10500`
4. 若需要双向 HTTPS（mTLS），在站点配置里启用 `ssl_client_certificate` 与 `ssl_verify_client on`。

## 验证命令

- 语法检查：`openresty -t` 或 `nginx -t`
- gRPC 健康检查：使用 TLS 客户端调用 `grpc.health.v1.Health/Check`
- 流式校验：调用 `com.continuousauth.proto.SensorDataService/StreamSensorData`

## 典型故障与原因

- `proxy_pass` 代替了 `grpc_pass`
  - 结果：OpenResty 按 HTTP/1.1 代理，gRPC 原生客户端无法正常上传数据。
- 上游协议写反：后端实际是 h2c，却配置成了 `grpcs://127.0.0.1:10500`
  - 结果：TLS 握手失败，常见报错是 upstream 握手或 preface 异常。
- `client_max_body_size` 没放宽
  - 结果：小请求能过，大一点的数据包失败；OpenResty 日志常见 `413` 或 `client intended to send too large body`。
- `grpc_read_timeout` / `grpc_send_timeout` 太短
  - 结果：`StreamSensorData` 建连后先正常，空闲一段时间或上传较慢时被代理层断开。
- 后端走 Docker bridge/端口映射，而上游是经域名/VIP 回源到本机
  - 结果：宿主机直跑可用，但容器化后可能只在本机/内网地址可用，经 VIP 路径回源异常；这类场景优先改用 `network_mode: host`。
- OpenResty 自己也跑在容器里，却把上游写成 `127.0.0.1:10500`
  - 结果：`127.0.0.1` 指向 OpenResty 容器自身，而不是宿主机。

## 建议排查顺序

1. 先直连宿主机端口验证后端本身：`grpcurl -plaintext 127.0.0.1:10500 grpc.health.v1.Health/Check`
2. 再经过 OpenResty 验证：`grpcurl -insecure <your-domain>:443 grpc.health.v1.Health/Check`
3. 查看 OpenResty 错误日志，重点看：
   - `client intended to send too large body`
   - `upstream sent no valid HTTP/2 connection preface`
   - `upstream timed out`
   - `upstream prematurely closed connection`
4. 确认当前服务端模式：
   - 当前推荐 compose 使用 `network_mode: host`，容器内 gRPC 直接监听宿主机 `10500`
   - 容器内 gRPC 默认是 h2c，而不是 TLS，除非显式配置了 `TLS_CERTFILE` / `TLS_KEYFILE`

本次仓库内联调按上面这套方式完成：OpenResty 对外使用 `443/TLS+h2`，上游回源到本项目默认的 `127.0.0.1:10500(h2c)`。
