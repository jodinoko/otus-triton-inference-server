# Логи Perf Analyzer

Модели тестировались по одному запросу за раз. Время указано в микросекундах: 1000 usec = 1 мс.

**Эмбеддер**

```text
Request concurrency: 1
  Client:
    Request count: 1214
    Throughput: 11.2399 infer/sec
    Avg latency: 88989 usec (standard deviation 28450 usec)
    p50 latency: 77877 usec
    p90 latency: 125894 usec
    p95 latency: 150127 usec
    p99 latency: 185344 usec
    Avg HTTP time: 88978 usec (send/recv 59 usec + response wait 88919 usec)
  Server:
    Inference count: 1214
    Execution count: 1214
    Successful request count: 1214
    Avg request latency: 88615 usec (overhead 6 usec + queue 42 usec + compute input 18 usec + compute infer 88480 usec + compute output 68 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 11.2399 infer/sec, latency 88989 usec
```

**Классификатор**

```text
Request concurrency: 1
  Client:
    Request count: 535847
    Throughput: 4954.4 infer/sec
    Avg latency: 201 usec (standard deviation 88 usec)
    p50 latency: 197 usec
    p90 latency: 226 usec
    p95 latency: 240 usec
    p99 latency: 287 usec
    Avg HTTP time: 199 usec (send/recv 21 usec + response wait 178 usec)
  Server:
    Inference count: 535846
    Execution count: 535846
    Successful request count: 535847
    Avg request latency: 54 usec (overhead 10 usec + queue 30 usec + compute input 3 usec + compute infer 7 usec + compute output 3 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 4954.4 infer/sec, latency 201 usec
```

**Генерация**

```text
Request concurrency: 1
  Client:
    Request count: 18
    Throughput: 0.166651 infer/sec
    Avg latency: 5765549 usec (standard deviation 381102 usec)
    p50 latency: 5704676 usec
    p90 latency: 6087177 usec
    p95 latency: 6409558 usec
    p99 latency: 6700208 usec
    Avg HTTP time: 5765540 usec (send/recv 54 usec + response wait 5765486 usec)
  Server:
    Inference count: 18
    Execution count: 18
    Successful request count: 18
    Avg request latency: 5765175 usec (overhead 5 usec + queue 39 usec + compute input 17 usec + compute infer 5765053 usec + compute output 60 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 0.166651 infer/sec, latency 5765549 usec
```

**BLS-пайплайн**

```text
Request concurrency: 1
  Client:
    Request count: 18
    Throughput: 0.16665 infer/sec
    Avg latency: 5900497 usec (standard deviation 633761 usec)
    p50 latency: 5919455 usec
    p90 latency: 6365999 usec
    p95 latency: 7189005 usec
    p99 latency: 7266473 usec
    Avg HTTP time: 5900486 usec (send/recv 52 usec + response wait 5900434 usec)
  Server:
    Inference count: 18
    Execution count: 18
    Successful request count: 18
    Avg request latency: 5900162 usec (overhead 1246 usec + queue 109 usec + compute 5898807 usec)

  Composing models:
  classifier, version: 1
      Inference count: 18
      Execution count: 18
      Successful request count: 18
      Avg request latency: 118 usec (overhead 11 usec + queue 41 usec + compute input 18 usec + compute infer 37 usec + compute output 10 usec)

  embeddings, version: 1
      Inference count: 18
      Execution count: 18
      Successful request count: 18
      Avg request latency: 71451 usec (overhead 4 usec + queue 33 usec + compute input 6 usec + compute infer 71371 usec + compute output 36 usec)

  generation, version: 1
      Inference count: 18
      Execution count: 18
      Successful request count: 18
      Avg request latency: 5827366 usec (overhead 4 usec + queue 35 usec + compute input 12 usec + compute infer 5827272 usec + compute output 42 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 0.16665 infer/sec, latency 5900497 usec
```
