FROM python:3.9 

RUN apt-get update && apt-get install -y
RUN useradd -m spydim 
RUN chown -R spydim:spydim /home/spydim  #spydim becomes the owner of this image
COPY --chown=spydim ./requirements.txt /home/spydim
USER spydim 
RUN cd /home/spydim && pip3 install -r requirements.txt
WORKDIR /home/spydim