from tkinter import *
import joblib
import pandas as pd

def predict_price():
    brand = str(brand_entry.get())
    model = str(model_entry.get())
    reg_date = float(reg_date_entry.get())
    km = float(km_entry.get())
    capacity = int(capacity_entry.get())
    vehicle_type = int(vehicle_type_entry.get())

    model_dt = joblib.load('models/decision_tree_model.pkl')
    encoded_columns = joblib.load('models/encoded_columns.pkl')

    df = pd.DataFrame({
        'Reg_Date': [reg_date],
        'Km': [km],
        'Capacity': [capacity],
        'Type': [vehicle_type],
        'Brand': [brand],
        'Model': [model]
    })

    categorical_columns = ['Brand', 'Model']
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower())
    encoded_data = pd.get_dummies(df, columns=categorical_columns)
    new_data_encoded = encoded_data.reindex(columns=encoded_columns, fill_value=0)

    result = model_dt.predict(new_data_encoded)

    result_label.config(text=f"Giá xe được đoán là {result[0]:,.03f} vnd")
    print(f"Giá xe được đoán là {result[0]:.03f} vnd")


if __name__ == '__main__':
    master = Tk()
    master.geometry("450x600")
    master.title("Dự đoán giá xe máy cũ sử dụng mô hình học máy")

    Label(master, text="Dự đoán giá xe máy đã qua sử dụng", bg="gray", fg="white",\
          font=("Helvetica", 14)).grid(row=0, columnspan=2, sticky="ew")

    labels = ["Hãng xe", "Dòng xe", "Năm đăng ký", "Đã đi (Km)", "Loại xe (0: Xe côn/moto, 1: Xe ga, 2: Xe số)",\
              "Capacity (cc) [0: 100-175, 1: 50-100, 2: >175]"]
    for i, label_text in enumerate(labels, start=1):
        Label(master, text=label_text, font=("Helvetica", 10)).grid(row=i, column=0, sticky="w", padx=(10, 0))

    brand_entry = Entry(master)
    model_entry = Entry(master)
    reg_date_entry = Entry(master)
    km_entry = Entry(master)
    capacity_entry = Entry(master)
    vehicle_type_entry = Entry(master)

    brand_entry.grid(row=1, column=1, sticky="w", padx=(0, 10))
    model_entry.grid(row=2, column=1, sticky="w", padx=(0, 10))
    reg_date_entry.grid(row=3, column=1, sticky="w", padx=(0, 10))
    km_entry.grid(row=4, column=1, sticky="w", padx=(0, 10))
    capacity_entry.grid(row=5, column=1, sticky="w", padx=(0, 10))
    vehicle_type_entry.grid(row=6, column=1, sticky="w", padx=(0, 10))

    master.columnconfigure(1, weight=1)

    result_label = Button(master, text="Dự đoán", command=predict_price)
    result_label.grid(row=7, columnspan=2, pady=(10, 0))

    for i in range(8):
        master.rowconfigure(i, weight=1)

    master.mainloop()
