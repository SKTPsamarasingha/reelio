import asyncio, ssl

REMOTE_HOST = "db.prisma.io"
REMOTE_PORT = 5432
LOCAL_PORT = 15433

async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()

async def handle_client(local_r, local_w):
    ctx = ssl.create_default_context()
    remote_r, remote_w = await asyncio.open_connection(
        REMOTE_HOST, REMOTE_PORT, ssl=ctx, server_hostname=REMOTE_HOST
    )
    await asyncio.gather(pipe(local_r, remote_w), pipe(remote_r, local_w))

async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", LOCAL_PORT)
    print(f"Prisma TLS SNI proxy listening on 127.0.0.1:{LOCAL_PORT} -> {REMOTE_HOST}:{REMOTE_PORT}")
    async with server:
        await server.serve_forever()

asyncio.run(main())
