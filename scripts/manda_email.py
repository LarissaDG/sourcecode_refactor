import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

REMETENTE   = "codigosecreto2025@gmail.com"
SENHA       = "bpij xclh mwqr sofe"
DESTINATARIO = "laladg18@gmail.com"


def enviar(assunto: str, corpo: str):
    msg = MIMEMultipart()
    msg["From"]    = REMETENTE
    msg["To"]      = DESTINATARIO
    msg["Subject"] = assunto
    msg.attach(MIMEText(corpo, "plain"))

    try:
        servidor = smtplib.SMTP("smtp.gmail.com", 587)
        servidor.starttls()
        servidor.login(REMETENTE, SENHA)
        servidor.sendmail(REMETENTE, DESTINATARIO, msg.as_string())
        servidor.quit()
        print(f"E-mail enviado: {assunto}")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")


if __name__ == "__main__":
    assunto = sys.argv[1] if len(sys.argv) > 1 else "Job finalizado"
    corpo   = sys.argv[2] if len(sys.argv) > 2 else "O job terminou de rodar."
    enviar(assunto, corpo)
