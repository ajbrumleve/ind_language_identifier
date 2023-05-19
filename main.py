import wx

import predict_language

try:
    import language_id
except MemoryError:
    pass


class MainPanel(wx.Panel):

    def __init__(self, parent):
        super().__init__(parent)
        input_label = wx.StaticText(self, label="Text to analyze")
        output_label = wx.StaticText(self, label="Predicted language")
        self.input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER, size=(400, 25))
        button = wx.Button(self, label='Predict')
        self.result_txt = wx.TextCtrl(self, size=(400, 100), style=wx.TE_MULTILINE)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(input_label, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer.Add(self.input, flag=wx.ALL | wx.CENTER, border=5)

        main_sizer.Add(button, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer.Add(output_label, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer.Add(self.result_txt, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        button.Bind(wx.EVT_BUTTON, self.predict)
        self.SetSizer(main_sizer)

    def predict(self, event):
        input_text = self.input.GetValue()
        prediction, familiarity, prob_dict = predict_language.predict(input_text)
        try:
            markov_prediction = language_id.process_input(input_text)
        except NameError:
            markov_prediction = "Markov memory error"
        self.result_txt.SetValue(f"{prediction} - Input text matches most likely language profile {familiarity * 100}%"
                                 f"\n\nThe probabilities for each language are: {prob_dict}\n\nThe Markov model "
                                 f"predicts {markov_prediction}")


class MyFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, title="Indonesian language predictor")
        panel = MainPanel(self)
        self.Show()


if __name__ == '__main__':
    app = wx.App(redirect=False)
    frame = MyFrame()
    app.MainLoop()
