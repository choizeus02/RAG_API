from flask import Flask, request, jsonify
from langchain.retrievers import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain_openai import ChatOpenAI
import json

app = Flask(__name__)

cpu_rank = {
    'AMD Ryzen Threadripper PRO 7995WX': 0,
    'AMD Ryzen Threadripper 7980X': 1,
    'AMD Ryzen Threadripper 7970X': 11,
    'AMD Ryzen Threadripper PRO 7975WX': 13,
    'AMD Ryzen Threadripper PRO 5995WX': 14,
    'AMD Ryzen Threadripper PRO 7965WX': 23,
    'AMD Ryzen Threadripper 7960X': 24,
    'Intel Core i9-14900KS': 56,
    'AMD Ryzen 9 7950X': 60,
    'AMD Ryzen 9 7950X3D': 63,
    'Intel Core i9-13900KS': 67,
    'Intel Core i7-14700K': 98,
    'AMD Ryzen 9 7900X': 107,
    'AMD Ryzen 9 7900X3D': 111,
    'Intel Core i7-13700K': 136,
    'Intel Core i5-14600KF': 181,
    'Intel Core i5-13600K': 194,
    'AMD Ryzen 7 7700X': 209,
    'AMD Ryzen 7 7800X3D': 225,
    'Intel Core i5-14500': 247,
    'Intel Core i5-13500': 257,
    'AMD Ryzen 5 7600X': 312,
    'Intel Core i5-12600K': 332,
    'Apple M3 Pro 12 Core': 335,
    'AMD Ryzen 5 7500F': 355,
    'AMD Ryzen 7 5700X': 359,
    'Intel Core i5-14400': 361,
    'AMD Ryzen 7 5700X3D': 366,
    'Intel Core i5-13400F': 399,
    'AMD Ryzen 5 5600X': 504,
    'Intel Core i5-12400F': 595,
    'Intel Core i3-12100E': 844
}

gpu_rank = {
    'GeForce RTX 4090': 0,
    'GeForce RTX 4080': 1,
    'GeForce RTX 4070 Ti': 4,
    'Radeon RX 7900 XTX': 6,
    'Radeon PRO W7800': 11,
    'RTX 6000 Ada Generation': 14,
    'Radeon RX 6800 XT': 25,
    'GeForce RTX 3070 Ti': 28,
    'GeForce RTX 4060 Ti 16GB': 31,
    'GeForce RTX 4060 Ti': 32,
    'NVIDIA A10': 37,
    'RTX A5500': 43,
    'GeForce RTX 3060 Ti': 47,
    'Radeon RX 6700 XT': 53,
    'Radeon RX 6750 GRE 12GB': 67,
    'Radeon PRO W7600': 72,
    'Radeon RX 6650 XT': 80,
    'GeForce RTX 3060 12GB': 81,
    'Radeon RX 6600 XT': 88,
    'GeForce RTX 3060 8GB': 106,
    'RTX A2000 12GB': 132,
    'GeForce GTX 1660 Ti': 152,
    'GeForce RTX 3050 8GB': 153,
    'GeForce RTX 3050 OEM': 170,
    'L4': 183,
    'Intel Arc A580': 184,
    'GeForce RTX 3050 6GB': 196,
    'GeForce GTX 1650 SUPER': 207
}

extended_examples = [
    # 가격대별 게이밍 PC
    (
        "200만원 이하의 게이밍 PC를 추천해줘.",
        {
            "query": "200만원 이하 게이밍 PC 견적",
            "filter": "and(gte('total_price', 1800000), lte('total_price', 2000000))"
        }
    ),
    (
        "100만원대의 사무용 컴퓨터 견적을 보여줘.",
        {
            "query": "사무용",
            "filter": "and(gte('total_price', 900000), lte('total_price', 1100000)"
        }
    ),
    (
        "300만원 이내의 고성능 그래픽 작업용 PC를 추천해줘.",
        {
            "query": "300만원이내, 그래픽 작업용",
            "filter": "and(gte('total_price', 2800000), lte('total_price', 3000000),eq('performance_grade','고성능'))"
        }
    ),
    # 특별한 조건
    (
        "예산 150만원으로 로스트아크를 풀옵션으로 돌릴 수 있는 PC를 추천해줘.",
        {
            "query": "로스트아크 ,로아 ,풀옵션 ",
            "filter": "and(gte('total_price', 1400000), lte('total_price', 1500000))"
        }
    ),
    (
        "3D 모델링 및 렌더링을 위한 PC 견적을 추천해줘",
        {
            "query": "3D 모델링 , 렌더링 ",
            "filter": None
        }
    ),
    (
        "롤과 배그가 돌아가는 PC 견적을 추천해줘",
        {
            "query": "롤,배그,게이밍",
            "filter": None
        }
    ),
    (
        "예산 100만원으로 사무용 PC를 추천해줘.",
        {
            "query": "사무용",
            "filter": "and(gte('total_price', 900000), lte('total_price', 1000000))"
        }
    ),
    (
        "예산 200만원 이하로 고성능 게이밍 PC를 추천해줘.",
        {
            "query": "게이밍",
            "filter": "and(lte('total_price', 2000000)),eq('performance_grade','고성능')"
        }
    ),
    (
        "예산 80만원으로 기본적인 인터넷 서핑과 문서 작업이 가능한 PC를 추천해줘.",
        {
            "query": "인터넷 서핑, 문서 작업",
            "filter": "and(gte('total_price', 700000), lte('total_price', 800000))"
        }
    ),
    (
        "예산 200만원으로 대학생이 사용하기 좋은 PC를 추천해줘.",
        {
            "query": "대학생",
            "filter": "and(gte('total_price', 1800000), lte('total_price', 2000000))"
        }
    ),
    (
        "예산 150만원으로 가성비 좋은 게이밍 PC를 추천해줘.",
        {
            "query": "가성비",
            "filter": "and(gte('total_price', 1400000), lte('total_price', 1500000))"
        }
    ),
    (
        "영상 편집과 게임을 동시에 할 수 있는 PC를 추천해줘.",
        {
            "query": "영상 편집, 게임",
            "filter": None
        }
    ),
    (
        "예산 120만원으로 집에서 영화 감상과 간단한 게임을 할 수 있는 PC를 추천해줘.",
        {
            "query": "영화 감상, 간단한 게임",
            "filter": "and(gte('total_price', 1100000), lte('total_price', 1200000))"
        }
    ),
    (
        "예산 300만원으로 프로그래밍과 디자인 작업을 동시에 할 수 있는 PC를 추천해줘.",
        {
            "query": "프로그래밍, 디자인 작업",
            "filter": "and(gte('total_price', 2800000), lte('total_price', 3000000))"
        }
    ),
    (
        "가격 200만원이상 롤,오버워치를 할수있으며 영상편집을 취미수준으로 할수있고 하드디스크 500GB이상인 pc를 추천해줘",
        {
            "query": "롤,오버워치,게이밍,영상편집,취미,하드디스크,HD,500GB이상",
            "filter": "and(gte('total_price', 2000000))"
        }
    ),
    (
        "그래픽카드 3060이상이고 cpu는 i7-11세대 이상이 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 3060, i7-11세대 이상",
            "filter": f"and(lte('gpu_score', {gpu_rank['GeForce RTX 3060 12GB']}), lte('cpu_score', {cpu_rank['Intel Core i7-13700K']}))"
        }
    ),
    # GPU 이하 조건 3개
    (
        "그래픽카드 3060 이하이고 cpu는 i5-12세대 이하가 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 3060, i5-12세대 이하",
            "filter": f"and(gte('gpu_score', {gpu_rank['GeForce RTX 3060 12GB']}), gte('cpu_score', {cpu_rank['Intel Core i5-12600K']}))"
        }
    ),
    # CPU 이상 조건 3개
    (
        "그래픽카드 3060 이상이고 cpu는 i7-12세대 이상이 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 3060, i7-12세대 이상",
            "filter": f"and(lte('gpu_score', {gpu_rank['GeForce RTX 3060 12GB']}), lte('cpu_score', {cpu_rank['Intel Core i7-13700K']}))"
        }
    ),
    # 하나는 이상이고 하나는 이하 조건 3개
    (
        "그래픽카드 3060 이상이고 cpu는 i5-12세대 이하가 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 3060, i5-12세대 이하",
            "filter": f"and(lte('gpu_score', {gpu_rank['GeForce RTX 3060 12GB']}), gte('cpu_score', {cpu_rank['Intel Core i5-12600K']}))"
        }
    ),
    (
        "그래픽카드 4080 이상이고 cpu는 i7-12세대 이하가 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 4080, i7-12세대 이하",
            "filter": f"and(lte('gpu_score', {gpu_rank['GeForce RTX 4080']}), gte('cpu_score', {cpu_rank['Intel Core i7-13700K']}))"
        }
    ),
    # GPU 이상 조건
    (
        "그래픽카드 3060 이상이고 cpu는 i7-11세대 이상이 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 3060, i7-11세대 이상",
            "filter": f"and(lte('gpu_score', {gpu_rank['GeForce RTX 3060 12GB']}), lte('cpu_score', {cpu_rank['Intel Core i7-13700K']}))"
        }
    ),
    # GPU 이하 조건 3개

    (
        "그래픽카드 3050 이하이고 cpu는 i5-12세대 이하가 들어갔으면 좋겠어 조건을 충족하는 pc 피씨를 추천해줘",
        {
            "query": "그래픽카드 3050, i5-12세대 이하",
            "filter": f"and(gte('gpu_score', {gpu_rank['GeForce RTX 3050 8GB']}), gte('cpu_score', {cpu_rank['Intel Core i5-12600K']}))"
        }
    ),

    # 두가지부품을 넣을때
    (
        "Intel Core i9-13900KS와 RTX 6000 Ada Generation이 들어간 PC를 추천해줘",
        {
            "query": "Intel Core i9-13900KS, RTX 6000 Ada Generation",
            "filter": f"and(eq('cpu_score', {cpu_rank['Intel Core i9-13900KS']}), eq('gpu_score', {gpu_rank['RTX 6000 Ada Generation']}))"
        }
    ),
    (
        "Intel Core i5-14600KF와 GeForce RTX 4060 Ti 16GB가 들어간 PC를 추천해줘",
        {
            "query": "Intel Core i5-14600KF, GeForce RTX 4060 Ti 16GB",
            "filter": f"and(eq('cpu_score', {cpu_rank['Intel Core i5-14600KF']}), eq('gpu_score', {gpu_rank['GeForce RTX 4060 Ti 16GB']}))"
        }
    ),
    # 컴퓨터 부품모든예외처리
    # 그래픽 처리
    (
        "GeForce RTX 4090을 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 4090 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 4090']})"
        }
    ),
    (
        "GeForce RTX 4080을 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 4080 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 4080']})"
        }
    ),
    (
        "GeForce RTX 4070 Ti를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 4070 Ti 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 4070 Ti']})"
        }
    ),
    (
        "Radeon RX 7900 XTX를 포함하는 PC를 추천해줘",
        {
            "query": "Radeon RX 7900 XTX 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon RX 7900 XTX']})"
        }
    ),
    (
        "RTX 6000 Ada Generation을 포함하는 PC를 추천해줘",
        {
            "query": "RTX 6000 Ada Generation 포함",
            "filter": f"eq('gpu_score', {gpu_rank['RTX 6000 Ada Generation']})"
        }
    ),
    (
        "Radeon RX 6800 XT를 포함하는 PC를 추천해줘",
        {
            "query": "Radeon RX 6800 XT 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon RX 6800 XT']})"
        }
    ),
    (
        "GeForce RTX 3070 Ti를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 3070 Ti 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 3070 Ti']})"
        }
    ),
    (
        "GeForce RTX 4060 Ti 16GB를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 4060 Ti 16GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 4060 Ti 16GB']})"
        }
    ),
    (
        "GeForce RTX 4060 Ti를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 4060 Ti 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 4060 Ti']})"
        }
    ),
    (
        "RTX A5500을 포함하는 PC를 추천해줘",
        {
            "query": "RTX A5500 포함",
            "filter": f"eq('gpu_score', {gpu_rank['RTX A5500']})"
        }
    ),
    (
        "GeForce RTX 3060 Ti를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 3060 Ti 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 3060 Ti']})"
        }
    ),
    (
        "Radeon RX 6700 XT를 포함하는 PC를 추천해줘",
        {
            "query": "Radeon RX 6700 XT 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon RX 6700 XT']})"
        }
    ),
    (
        "Radeon RX 6750 GRE 12GB를 포함하는 PC를 추천해줘",
        {
            "query": "Radeon RX 6750 GRE 12GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon RX 6750 GRE 12GB']})"
        }
    ),
    (
        "Radeon PRO W7600을 포함하는 PC를 추천해줘",
        {
            "query": "Radeon PRO W7600 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon PRO W7600']})"
        }
    ),
    (
        "Radeon RX 6650 XT를 포함하는 PC를 추천해줘",
        {
            "query": "Radeon RX 6650 XT 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon RX 6650 XT']})"
        }
    ),
    (
        "GeForce RTX 3060 12GB를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 3060 12GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 3060 12GB']})"
        }
    ),
    (
        "Radeon RX 6600 XT를 포함하는 PC를 추천해줘",
        {
            "query": "Radeon RX 6600 XT 포함",
            "filter": f"eq('gpu_score', {gpu_rank['Radeon RX 6600 XT']})"
        }
    ),
    (
        "GeForce RTX 3060 8GB를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 3060 8GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 3060 8GB']})"
        }
    ),
    (
        "RTX A2000 12GB를 포함하는 PC를 추천해줘",
        {
            "query": "RTX A2000 12GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['RTX A2000 12GB']})"
        }
    ),
    (
        "GeForce GTX 1660 Ti를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce GTX 1660 Ti 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce GTX 1660 Ti']})"
        }
    ),
    (
        "GeForce RTX 3050 8GB를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 3050 8GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 3050 8GB']})"
        }
    ),
    (
        "GeForce RTX 3050 6GB를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce RTX 3050 6GB 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce RTX 3050 6GB']})"
        }
    ),
    (
        "GeForce GTX 1650 SUPER를 포함하는 PC를 추천해줘",
        {
            "query": "GeForce GTX 1650 SUPER 포함",
            "filter": f"eq('gpu_score', {gpu_rank['GeForce GTX 1650 SUPER']})"
        }
    )
    # cpu
    , (
        "AMD Ryzen Threadripper PRO 7995WX를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen Threadripper PRO 7995WX 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen Threadripper PRO 7995WX']})"
        }
    ),
    (
        "AMD Ryzen Threadripper 7980X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen Threadripper 7980X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen Threadripper 7980X']})"
        }
    ),
    (
        "AMD Ryzen Threadripper 7970X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen Threadripper 7970X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen Threadripper 7970X']})"
        }
    ),
    (
        "AMD Ryzen Threadripper PRO 7975WX를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen Threadripper PRO 7975WX 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen Threadripper PRO 7975WX']})"
        }
    ),
    (
        "AMD Ryzen Threadripper PRO 5995WX를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen Threadripper PRO 5995WX 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen Threadripper PRO 5995WX']})"
        }
    ),
    (
        "AMD Ryzen Threadripper 7960X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen Threadripper 7960X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen Threadripper 7960X']})"
        }
    ),
    (
        "Intel Core i9-14900KS를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i9-14900KS 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i9-14900KS']})"
        }
    ),
    (
        "AMD Ryzen 9 7950X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 9 7950X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 9 7950X']})"
        }
    ),
    (
        "Intel Core i9-13900KS를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i9-13900KS 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i9-13900KS']})"
        }
    ),
    (
        "Intel Core i7-14700K를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i7-14700K 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i7-14700K']})"
        }
    ),
    (
        "AMD Ryzen 9 7900X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 9 7900X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 9 7900X']})"
        }
    ),
    (
        "AMD Ryzen 9 7900X3D를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 9 7900X3D 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 9 7900X3D']})"
        }
    ),
    (
        "Intel Core i7-13700K를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i7-13700K 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i7-13700K']})"
        }
    ),
    (
        "Intel Core i5-14600KF를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-14600KF 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-14600KF']})"
        }
    ),
    (
        "Intel Core i5-13600K를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-13600K 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-13600K']})"
        }
    ),
    (
        "AMD Ryzen 7 7700X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 7 7700X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 7 7700X']})"
        }
    ),
    (
        "AMD Ryzen 7 7800X3D를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 7 7800X3D 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 7 7800X3D']})"
        }
    ),
    (
        "Intel Core i5-14500를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-14500 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-14500']})"
        }
    ),
    (
        "Intel Core i5-13500를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-13500 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-13500']})"
        }
    ),
    (
        "AMD Ryzen 5 7600X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 5 7600X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 5 7600X']})"
        }
    ),
    (
        "Intel Core i5-12600K를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-12600K 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-12600K']})"
        }
    ),
    (
        "AMD Ryzen 5 7500F를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 5 7500F 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 5 7500F']})"
        }
    ),
    (
        "AMD Ryzen 7 5700X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 7 5700X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 7 5700X']})"
        }
    ),
    (
        "Intel Core i5-14400를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-14400 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-14400']})"
        }
    ),
    (
        "AMD Ryzen 7 5700X3D를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 7 5700X3D 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 7 5700X3D']})"
        }
    ),
    (
        "Intel Core i5-13400F를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-13400F 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-13400F']})"
        }
    ),
    (
        "AMD Ryzen 5 5600X를 포함하는 PC를 추천해줘",
        {
            "query": "AMD Ryzen 5 5600X 포함",
            "filter": f"eq('cpu_score', {cpu_rank['AMD Ryzen 5 5600X']})"
        }
    ),
    (
        "Intel Core i5-12400F를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i5-12400F 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i5-12400F']})"
        }
    ),
    (
        "Intel Core i3-12100E를 포함하는 PC를 추천해줘",
        {
            "query": "Intel Core i3-12100E 포함",
            "filter": f"eq('cpu_score', {cpu_rank['Intel Core i3-12100E']})"
        }
    )

]

attribute_info = [
    {"name": "quote_title", "description": "Title of the quote for the computer", "type": "string"},
    {"name": "date_create", "description": "Date the quote was created", "type": "date"},
    {"name": "cpu_gpu_combinations", "description": "Combination of CPU and GPU used in the computer",
     "type": "string"},
    {"name": "quotation_summary", "description": "Summary of the quotation", "type": "string"},
    {"name": "quote_person_introduction", "description": "Introduction of the person giving the quote",
     "type": "string"},
    {"name": "quote_feedback", "description": "Feedback given on the quote", "type": "string"},
    {"name": "computer_estimate_data", "description": "Data on the estimate of the computer", "type": "string"},
    {"name": "quote_description", "description": "Description of the quote", "type": "string"},
    {"name": "parts_price", "description": "Price of the parts of the computer", "type": "object"},
    {"name": "total_price", "description": "Total price of the computer", "type": "integer"},
    {"name": "CPU", "description": "CPU used in the computer", "type": "string"},
    {"name": "Motherboard", "description": "Motherboard used in the computer", "type": "string"},
    {"name": "Graphic Card", "description": "Graphic card used in the computer", "type": "string"},
    {"name": "SSD", "description": "SSD used in the computer", "type": "string"},
    {"name": "Memory", "description": "Memory used in the computer", "type": "string"},
    {"name": "Power Supply", "description": "Power supply used in the computer", "type": "string"},
    {"name": "CPU Cooler", "description": "CPU cooler used in the computer", "type": "string"},
    {'name': 'cpu_score',
     'description': 'Performance score of the CPU. Valid values are [0, 1, 11, 13, 14, 23, 24, 56, 60, 63, 67, 98, 107, 111, 136, 181, 194, 209, 225, 247, 257, 312, 332, 335, 355, 359, 361, 366, 399, 504, 595, 844]',
     'type': 'integer'},
    {'name': 'gpu_score',
     'description': 'Performance score of the GPU. Valid values are [0, 1, 4, 6, 11, 14, 25, 28, 31, 32, 37, 43, 47, 53, 67, 72, 80, 81, 88, 106, 132, 152, 153, 170, 183, 184, 196, 207]',
     'type': 'integer'},
    {'name': 'performance_grade',
     'description': "Overall performance grade of the computer. Valid values are ['고성능', '저성능', '중성능']",
     'type': 'string'},
    {'name': 'cpu_benchmarkscore','description': 'Overall benchmark score of the CPU based on performance tests. This score is typically a higher number indicating better performance. Valid values are integers.','type': 'integer'},
    {'name': 'gpu_benchmarkscore','description': 'Overall benchmark score of the GPU based on performance tests. This score is typically a higher number indicating better performance. Valid values are integers.','type': 'integer'}
]


def initialize_resources():
    OPENAI_KEY = ""
    doc_contents = "견적에 대한 자세한 설명"

    chain = load_query_constructor_runnable(
        ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_KEY),
        doc_contents,
        attribute_info,
        examples=extended_examples,
    )

    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db3 = Chroma(persist_directory="vectorstore_db1", embedding_function=embeddings)
    num_docs = db3._collection.count()
    print("Number of documents in vectorstore:", num_docs)

    return chain, db3

# 미리 초기화된 리소스
chain, db3 = initialize_resources()

def call_by_rag(text):
    if "추천" in text:
        retriever = SelfQueryRetriever(
            query_constructor=chain, vectorstore=db3, verbose=True, k=4
        )

        results = retriever.get_relevant_documents(text)
        return results

    print("다시 질문하시오")
    return []

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = call_by_rag(text)
    response = []
    for res in results:
        result_dict = json.loads(res.page_content)
        response.append(result_dict)

    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the PC Recommendation API!"

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
