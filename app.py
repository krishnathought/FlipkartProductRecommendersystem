from flask import render_template,Flask,request,Response
from prometheus_client import Counter,generate_latest
from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder

from dotenv import load_dotenv
load_dotenv()

REQUEST_COUNT = Counter("http_requests_total" , "Total HTTP Request")

def create_app():

    app = Flask(__name__)

    vector_store = DataIngestor().ingest(load_existing=True)
    builder = RAGChainBuilder(vector_store)   # ✅ single builder
    rag_chain = builder.build_chain() 
    
    session_id = "krishna"
    
    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")
    
    @app.route("/get" , methods=["POST"])
    def get_response():
        user_input = request.form["msg"]

        result = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(builder.get_history("krishna"))


        # If result is dict, try to extract "answer"
        if isinstance(result, dict) and "answer" in result:
            response_text = result["answer"]
        else:
            response_text = str(result)
        rows = [row.strip() for row in response_text.strip().split("\n") if row.strip()]

        table_html = "<table border='1' style='border-collapse: collapse; text-align: left;'>"

        for i, row in enumerate(rows):
            # Expecting columns separated by '|'
            cols = [col.strip() for col in row.split("|") if col.strip()]
            table_html += "<tr>"
            for col in cols:
                if i == 0:  # header row
                    table_html += f"<th style='padding:8px'>{col}</th>"
                else:
                    table_html += f"<td style='padding:8px'>{col}</td>"
            table_html += "</tr>"

            table_html += "</table>"

        return table_html
            

        #return response_text

        
    
    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")
    
    return app

if __name__=="__main__":
    app = create_app()
    app.run(host="0.0.0.0",port=5003,debug=True)