import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# === CONFIGURACIÓN DE CRIDENCIALES ===
# Reemplaza con tu clave real de OpenAI para las pruebas del taller
os.environ["OPENAI_API_KEY"] = "sk-proj-TU_API_KEY_AQUI"

# === 1. DEFINICIÓN DE LA HERRAMIENTA AUTÓNOMA (IE1) ===
def consultar_base_conocimiento(query: str) -> str:
    """Busca soluciones técnicas dentro de la base de datos de la organización."""
    base_datos_ti = {
        "error de clave": "El usuario debe ingresar a portal.duoc.cl para restablecer su contraseña de forma autónoma.",
        "pantalla negra": "Verificar que el cable de video (HDMI/VGA) esté conectado firmemente a la tarjeta gráfica dedicada.",
        "sin internet": "Comprobar si el cable de red Ethernet está conectado o reiniciar el adaptador de red Wi-Fi."
    }
    
    # Búsqueda simple de palabras clave dentro del diccionario
    for clave, solucion in base_datos_ti.items():
        if clave in query.lower():
            return solucion
            
    return "Materia no encontrada en la base de conocimientos. Se requiere escalar al equipo de Soporte Nivel 2."
# Declaración formal de la herramienta para el agente
herramientas = [
    Tool(
        name="ConsultarBaseConocimiento",
        func=consultar_base_conocimiento,
        description="Útil cuando necesitas buscar respuestas a problemas técnicos de TI en la empresa."
    )
]

# === 2. ORQUESTACIÓN DEL AGENTE Y PLANIFICACIÓN (IE2, IE5) ===
# Definimos el modelo base (un LLM rápido y económico)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Diseñamos el prompt del sistema que guía el razonamiento
prompt_sistema = ChatPromptTemplate.from_messages([
    ("system", "Eres un agente autónomo de soporte técnico de TI corporativo. Tu objetivo es resolver el problema del usuario de forma eficiente. Analiza el requerimiento, decide si debes usar la herramienta 'ConsultarBaseConocimiento' y genera una respuesta. Si no encuentras la solución, debes tomar la decisión de escalar el caso internamente."),
    MessagesPlaceholder(variable_name="historial_conversacion"), # Espacio para la memoria de contexto (IE3)
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Espacio donde el agente planifica sus pasos lógicos
])

# Construcción formal del agente basado en funciones
agente = create_openai_functions_agent(llm, herramientas, prompt_sistema)
ejecutor_agente = AgentExecutor(agent=agente, tools=herramientas, verbose=True) # El verbose=True muestra el pensamiento en consola

# === 3. CONFIGURACIÓN DE MEMORIA DE SESIÓN (IE3) ===
historial_mensajes = ChatMessageHistory()

agente_final = RunnableWithMessageHistory(
    ejecutor_agente,
    lambda session_id: historial_mensajes,
    input_messages_key="input",
    history_messages_key="historial_conversacion",
)

# === 4. SIMULACIÓN DE PRUEBAS DE ENTORNO (IE6) ===
# ID único para simular la sesión de un usuario de la empresa
configuracion_sesion = {"configurable": {"session_id": "ticket_usuario_01"}}

print("\n=== INTERACCIÓN 1: El agente buscará en la base de datos ===")
flujo_1 = agente_final.invoke(
    {"input": "Hola, mi pantalla se quedó negra y no da video. ¿Qué puedo hacer?"}, 
    config=configuracion_sesion
)
print("Respuesta del Agente:", flujo_1["output"])

print("\n=== INTERACCIÓN 2: El agente usa la memoria para continuar el contexto ===")
flujo_2 = agente_final.invoke(
    {"input": "Ya revisé el cable HDMI como me dijiste y sigue igual de negra. ¿Cuál es el siguiente paso?"}, 
    config=configuracion_sesion
)
print("Respuesta del Agente:", flujo_2["output"])