from spatiotemporaldataproc4 import *
import smtplib, os
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders

# This will log you into your gmail account--this is where the mail will be sent from.
gmail_user = sys.argv[1] # String e.g. mypassword
gmail_pwd = sys.argv[2] # String e.g. Password

# The parameters
to = "aszhang95@gmail.com"
subject = "[ZeD Lab - python autogenerated email] Datasets Finished"
text = "Script has finished running. :)"

def mail(to, subject, text):
    msg = MIMEMultipart()

    msg['From'] = gmail_user
    msg['To'] = to
    msg['Subject'] = subject

    msg.attach(MIMEText(text))
    mailServer =smtplib.SMTP("smtp.gmail.com", 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(gmail_user, gmail_pwd)
    mailServer.sendmail(gmail_user, to, msg.as_string())
    mailServer.close()

types1 = ["HOMICIDE", "ASSAULT", "BATTERY"]
types2 = ["THEFT", "MOTOR VEHICLE THEFT", "ROBBERY", "BURGLARY"]

proc = SpatialTemporalData(path="Crimes2001-2017.csv", bin_path="./bin/procdata")

proc.transform_with_binary(grid_size=200, force=True, type_list=types1, \
export_df_name="../../../../../../project2/ishanu/CRIME/data/TS1_200.csv")
proc.export(path="../../../../../../project2/ishanu/CRIME/data/TS1_200_dict.p")

proc.transform_with_binary(grid_size=250, type_list=types1, \
export_df_name="../../../../../../project2/ishanu/CRIME/data/TS1_250.csv")
proc.export(path="../../../../../../project2/ishanu/CRIME/data/TS1_250_dict.p")

proc.transform_with_binary(grid_size=200, force=True, type_list=types2, \
export_df_name="../../../../../../project2/ishanu/CRIME/data/TS2_200.csv")
proc.export(path="../../../../../../project2/ishanu/CRIME/data/TS2_200_dict.p")

proc.transform_with_binary(grid_size=250, type_list=types2, \
export_df_name="../../../../../../project2/ishanu/CRIME/data/TS2_250.csv")
proc.export(path="../../../../../../project2/ishanu/CRIME/data/TS2_250_dict.p")

mail(to, subject, text)
