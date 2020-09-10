FROM hamelsmu/fastpages-nbdev

WORKDIR /fastpages
COPY . .
RUN chmod u+x action_entrypoint.sh
RUN chmod u+x word2post.sh
RUN dos2unix /fastpages/*.sh
RUN pip install pandas

CMD [ "/fastpages/action_entrypoint.sh" ]
