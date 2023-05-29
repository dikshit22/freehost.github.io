import streamlit as st
import pickle

D_model = pickle.load(open('D_model.pkl', 'rb'))
A_model = pickle.load(open('A_model.pkl', 'rb'))
S_model = pickle.load(open('S_model.pkl', 'rb'))

def predict_class(answer):
    D_colind = [2, 4, 9, 12, 15, 16, 20]
    A_colind = [1, 3, 6, 8, 14, 18, 19]
    S_colind = [0, 5, 7, 10, 11, 13, 17]
    D_input = []
    A_input = []
    S_input = []
    for i in D_colind:
        D_input.append(answer[i])
    for i in A_colind:
        A_input.append(answer[i])
    for i in S_colind:
        S_input.append(answer[i])
    pred = [(D_model.predict([D_input]))[0], (A_model.predict([A_input]))[0], (S_model.predict([S_input]))[0]]
    return pred

def main():
    st.title("DAS(Depression, Anxiety and Stress) Screening Test")
    html_temp = '''
                <div style="border-width: 3px; border-style: solid; padding: 6px; border-radius: 6px;">
                    <h4>Please read each statement and give a number 0, 1, 2 or 3 that indicates how much the statement
                        applied to you <i><u>over the past week</u></i>. There are no right or wrong answers. Do not spend too much time
                        on any statement.
                    </h4>
                    <p style="font-size: 18px; line-height:150%;"> The rating scale is as follows:<br>
                        0 &nbsp;Did not apply to me at all<br>
                        1 &nbsp;Applied to me to some degree, or some of the time<br>
                        2 &nbsp;Applied to me to a considerable degree, or a good part of time<br>
                        3 &nbsp;Applied to me very much, or most of the time<br>
                    </p>
                </div>
                '''
    st.markdown(html_temp, unsafe_allow_html=True)
    with st.form("myform", clear_on_submit=True):
        answer = []
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 1. I find it hard to wind down.</b></font>''', unsafe_allow_html=True)
        answer.append(st.slider('1', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 2. I was aware of dryness of my mouth.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('2', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 3. I could not seem to experience any positive feeling at all.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('3', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 4. I experienced breathing difficulty.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('4', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 5. I find it difficult to work up the initiative to do things.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('5', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 6. I tend to over-react to situations.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('6', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 7. I experienced trembling (eg, in the hands).</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('7', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 8. I feel that I was using a lot of nervous energy.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('8', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b> 9. I was worried about situations in which I might panic and make a fool of myself.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('9', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>10. I felt that I had nothing to look forward to.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('10', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>11. I find myself getting agitated.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('11', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>12. I find it is difficult to relax.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('12', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>13. I felt down-hearted and blue.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('13', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>14. I was intolerant of anything that kept me from getting on with what I was doing.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('14', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>15. I felt I was close to panic.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('15', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>16. I was unable to become enthusiastic about anything.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('16', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>17. I felt I was not worth much as a person.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('17', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>18. I feel that I was rather touchy.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('18', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>19. I was aware of the action of my heart in the absence of physical exertion.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('19', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>20. I feel scared without any good reason.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('20', min_value=0, max_value=3, label_visibility='collapsed'))
        st.markdown(
            '''<font face='Times New Roman' size='4'><b>21. I feel that life is meaningless.</b></font>''',
            unsafe_allow_html=True)
        answer.append(st.slider('21', min_value=0, max_value=3, label_visibility='collapsed'))


        if st.form_submit_button("Predict", type="primary"):
            ans = predict_class(answer)
            state = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
            color = ['Green', 'Yellow', 'Orange', 'Red', 'Brown']
            html1 = f'''<div align="center"><div align="center" style="padding: 6px; border-radius: 6px; width: 300px; background: {color[ans[0]]}; color: black;">{state[ans[0]]} Depression</div></div><br>'''
            html2 = f'''<div align="center"><div align="center" style="padding: 6px; border-radius: 6px; width: 300px; background: {color[ans[1]]}; color: black;">{state[ans[1]]} Anxiety</div></div><br>'''
            html3 = f'''<div align="center"><div align="center" style="padding: 6px; border-radius: 6px; width: 300px; background: {color[ans[2]]}; color: black;">{state[ans[2]]} Stress</div></div><br>'''
            st.markdown(html1, unsafe_allow_html=True)
            st.markdown(html2, unsafe_allow_html=True)
            st.markdown(html3, unsafe_allow_html=True)
            # st.success(state[ans[0]], 'Depression')
            # st.success(state[ans[1]])
            # st.success(state[ans[2]])


if __name__ == '__main__':
    main()