import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="localhost")
)

channel = connection.channel()

channel.queue_declare(queue="summarize_queue")

print("RabbitMQ ket noi thanh cong!")
print("Queue summarize_queue da duoc tao.")

connection.close()