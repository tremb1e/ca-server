
---

# **Continuous Authentication - 服务端处理程序功能规格（Lite v1.0）**
> **版本**：Lite v1.0  
> **最后更新日期**：2025-09-16  
> **状态**：功能规格说明书（MVP 最小实现）  
> **目标**：与 Android App 完全功能兼容，专注核心数据接收存储链路。

---

## **1. 项目简介**
服务端处理程序用于接收、解密、校验并存储来自移动端的加密传感器数据包。  
在 Lite/MVP 阶段，不使用数据库，而是采用**纯文件存储**，**按设备 ID 哈希建目录**，**每个 Session 汇总为单个文件**，并按包序号顺序合并。

---

## **2. 系统架构（MVP版）**
```
[App端] => HTTPS POST (binary_encrypted_envelope)
          ↓
[服务器API入口] (/api/v1/sensor-data)
          ↓
[请求头解析] -> [路径规划: hashid/session_id]
          ↓
[数据解密(AES-256-GCM固定密钥)]
          ↓
[LZ4解压缩]
          ↓
[数据包 JSON 解析 & 校验]
          ↓
[按 session 文件顺序追加保存]
          ↓
[响应 200 OK]
```

---

## **3. 核心模块**
### **3.1 HTTP 接口模块**
- **依赖**：Python FastAPI 
- **接口**：  
  ```
  POST /api/v1/sensor-data
  Content-Type: application/octet-stream
  Headers:
    X-Device-ID-Hash: <device_id_hash>
    X-Session-ID: <session_id>
    X-Packet-Sequence: <int>
  Body: binary_encrypted_envelope
  ```
- **约束**：
  - `device_id_hash`：目录名
  - `session_id`：子目录名或直接作为文件名一部分
  - `packet_seq_no`：写入时按顺序追加
  - 上层由 App 保证按 packet_seq_no 顺序发送，服务端检查是否乱序，需要按时间顺序写入文件中

---

### **3.2 数据解密模块**
- **算法**：AES-256-GCM（与 App 端一致）
- **密钥管理**：
  - MVP 阶段使用 **固定对称密钥**，密钥为“Continuous_Authentication”
- **解密输入**：二进制包（IV + Ciphertext + GCM Tag）
- **输出**：UTF-8 JSON 文本

---

### **3.3 数据校验模块**
- 校验字段：
  - `session_id` 与请求头 X-Session-ID 一致
  - `packet_seq_no` 与请求头一致
  - `device_id_hash` 与请求头一致
  - 时间戳为有效 UTC 毫秒值
- 格式校验：
  - 传感器数组字段存在
  - 数值类型正确

---

### **3.4 存储模块（文件系统）**
- **目录结构**：
  ```
  data_root/
      ├── <device_id_hash>/
      │      ├── session_<session_id>.jsonl
      │      └── ...
      └── ...
  ```
- **写入规则**：
  - 每个设备一个文件夹
  - 每个 session 对应 1 个 `.jsonl` 文件
  - 来包即 **按行追加** JSON 文本（一个包一行）
  - 包含 `packet_seq_no` 方便后续排序
- **文件格式**：
  - **JSON Lines（.jsonl）**：便于流式读取与追加
  - 原始 JSON 不作额外加工（保留完整字段）

---

### **3.5 错误处理 & 响应模块**
- 成功：
  ```json
  { "status": "ok" }
  ```
- 解密失败：
  ```json
  { "status": "error", "reason": "decryption_failed" }
  ```
- JSON 格式错误：
  ```json
  { "status": "error", "reason": "invalid_json" }
  ```
- 校验不通过：
  ```json
  { "status": "error", "reason": "validation_failed" }
  ```

---

## **4. 接口详细规格**
### **4.1 POST /api/v1/sensor-data**
#### **请求**
- **Headers**：
  - `Content-Type: application/octet-stream`
  - `X-Device-ID-Hash`: String(32+)
  - `X-Session-ID`: String
  - `X-Packet-Sequence`: Integer
- **Body**：
  - 二进制 AES-GCM 加密包

#### **解密后的 JSON 示例**
```json
{
  "device_id_hash": 123456789,
  "session_id": 123456789,
  "packet_seq_no": 123456789,
  "timestamp_ms": 1643723400000,
  "window_start_ms": 1643723400000,
  "window_end_ms": 1643723401000,
  "type": "sensor",
  "foreground_package_name": "包名",
  "sensor_data": [
    {
      "sensor_name": "accelerometer",
      "sensor_type": 1,
      "timestamp_ns": 123456789,
      "values": { "x": 0.1, "y": 0.2, "z": 9.8 },
      "accuracy": 3
    }
  ]
}
```

#### **响应**
```json
{ "status": "ok" }
```

---

## **5. 运行配置**
1.编译后使用docker部署运行
2.将存储和日志挂载出来

---

## **6. 日志系统**
使用完备的日志系统详细记录系统运行信息，连接设备相关信息，加解密情况，数据整合等等内容。

