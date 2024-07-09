import streamlit as st

st.set_page_config(
    page_title="Amazon Reviews for Sentiment Analysis",
    page_icon="img/mzn.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image("img/ath2.png")

st.title('Dashboard Amazon Reviews for Sentiment Analysis')
st.caption('Analyze of consumer responses on Amazon e-commerce platform')
st.markdown("<br>", unsafe_allow_html=True)

st.write('Perkembangan teknologi informasi dan komunikasi telah mendorong pertumbuhan pesat e-commerce, yang kini menjadi salah satu sektor bisnis yang paling dinamis dan menguntungkan di dunia. Amazon, pelopor e-commerce, telah memainkan peran penting dalam mentransformasi industri ritel dan membentuk lanskap e-commerce global. Kesuksesan Amazon dapat disebabkan oleh berbagai faktor, termasuk strategi bisnis yang inovatif, fokus pada kepuasan pelanggan, dan infrastruktur teknologi yang canggih. Amazon telah menjadi model bagi banyak perusahaan e-commerce lainnya, dan keberhasilannya telah menginspirasi munculnya berbagai platform e-commerce yang bersaing di pasar global.')

with st.columns(3)[1]:
    st.image("img/review e-com.png")

st.write('Industri E-commerce Amazon telah berkembang pesat dan menjadi salah satu pemain utama di pasar e-commerce global. Namun, pertumbuhan yang sukses ini juga disertai dengan tantangan-tantangan yang signifikan dalam mengelola ulasan produk pelanggan. Dengan jumlah ulasan yang sangat besar yang diterima oleh Amazon, mengelola dan memahami ulasan tersebut telah menjadi tugas yang sangat sulit dan kompleks.')
st.write('Proyek ini dirancang untuk menggantikan metode manual yang telah lama digunakan dalam proses perusahaan. Selain itu, proyek ini adalah investasi strategis yang ditujukan untuk meningkatkan efisiensi operasional, memungkinkan pengambilan keputusan yang lebih bijaksana, mengurangi kerugian  dan meningkatkan pengalaman pelanggan.')
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Intangible Benefits")
st.write("- Berpotensi peningkatan penjualan dan loyalitas pelanggan.")
st.write("- Penghematan biaya operasional dalam pengelolaan ulasan pelanggan secara manual.")
st.write("- Memahami lebih dalam pendapat dan emosi pelanggan.")
st.write("- Penyempurnaan produk berdasarkan ulasan pelanggan. ")
st.write("- Memungkinkan perusahaan menangani masalah pelanggan dengan cepat.")
st.write("- Pengambilan keputusan strategis berdasarkan data pelanggan.")
st.write("- Penanganan masalah pelanggan dengan cepat.")
st.write("- Inovasi produk yang lebih efektif.")
st.write("- Menjaga relevansi dan kesetiaan pelanggan jangka panjang.")
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Outline")
st.markdown("<br>", unsafe_allow_html=True)
st.image("img/proses.png",width=900)