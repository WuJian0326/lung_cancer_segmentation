
	
	check_point 		儲存model的資料夾
	
	
	models			各種模型存放



	SEG_Train_Datasets	訓練、測試資料

		Mix_858_471		混合處理影像(縮放、合併)
			Train_Image	訓練輸入
			Train_Mask	訓練groundtruth

		Origial_image	未做任何處理影像
			Train_Annotations 訓練groundtruth json文件
			Train_Image	訓練輸入
			Train_Mask	訓練groundtruth
			create_mask.py 將Annotations轉成Train_Mask

		Test			private public 測試資料
			output_mask	測試輸出結果
			private	private原始影像
			private_input_471_858	resize後的private影像
			public	public原始影像
			public_input_471_858	resize後的public影像
			trainset	訓練資料抽樣資料 原始影像
			trainset_input_471_858	resize後的trainset影像
			traintarget	訓練資料抽樣資料的groundtruth
			output_process.py	對Test中的資料做縮放	
		Image_Process.py	resize圖像
		Mix_process.py	創建混合影像
		random_image.py	未使用

	
DataLoader.py		讀取影像程式
F1_score.py			計算F1_score
Image_Process.py		影像前處理函數
loss_fn.py			自定義Loss function
train.py			訓練主函數
trainer.py			訓練過程class
utils_tool.py		儲存模型
