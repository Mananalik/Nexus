import {
  Image,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import { useRouter } from "expo-router";
import { Formik } from "formik";
import { SafeAreaView } from "react-native-safe-area-context";
import logo from "../../assets/images/NexusLogo.png";
import authSchema from "../../utils/authSchema";

const Signup = () => {
  const router = useRouter();
  const handleSignup = () => {};
  return (
    <SafeAreaView className="bg-[#030303]">
      <ScrollView contentContainerStyle={{ height: "130%" }}>
        <View className="flex justify-center items-center px-6 py-10">
          <View className="flex flex-col items-center mb-12">
            <Image source={logo} style={{ height: 150, width: 150 }} />
            <Text className="text-[#F1EFEC] text-5xl font-extrabold tracking-widest">
              Nexus
            </Text>
            <Text className="text-[#D4C9BE] text-base mt-2">
              Your finances, connected.
            </Text>
          </View>

          <Text className="text-[#F1EFEC] text-3xl font-semibold mb-6">
            Sign Up
          </Text>

          <View className="w-full max-w-md">
            <Formik
              initialValues={{
                name: "",
                number: "",
                email: "",
                password: "",
              }}
              onSubmit={handleSignup}
              validationSchema={authSchema}
            >
              {({
                handleChange,
                handleBlur,
                values,
                errors,
                touched,
                handleSubmit,
              }) => (
                <View className="space-y-5">
                  <View>
                    <Text className="text-[#F1EFEC] mb-2 font-medium">
                      Full Name
                    </Text>
                    <TextInput
                      className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                      onChangeText={handleChange("name")}
                      onBlur={handleBlur("name")}
                      value={values.name}
                    />
                    {touched.name && errors.name && (
                      <Text className="text-red-500 text-xs mt-1">
                        {errors.name}
                      </Text>
                    )}
                  </View>

                  <View>
                    <Text className="text-[#F1EFEC] my-2 font-medium">
                      Phone Number
                    </Text>
                    <TextInput
                      className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                      onChangeText={handleChange("number")}
                      onBlur={handleBlur("number")}
                      value={values.number}
                      keyboardType="phone-pad"
                    />
                    {touched.number && errors.number && (
                      <Text className="text-red-500 text-xs mt-1">
                        {errors.number}
                      </Text>
                    )}
                  </View>

                  <View>
                    <Text className="text-[#F1EFEC] my-2 font-medium">
                      Email
                    </Text>
                    <TextInput
                      className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                      onChangeText={handleChange("email")}
                      onBlur={handleBlur("email")}
                      value={values.email}
                      keyboardType="email-address"
                    />
                    {touched.email && errors.email && (
                      <Text className="text-red-500 text-xs mt-1">
                        {errors.email}
                      </Text>
                    )}
                  </View>

                  <View>
                    <Text className="text-[#F1EFEC] my-2 font-medium">
                      Password
                    </Text>
                    <TextInput
                      className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                      onChangeText={handleChange("password")}
                      onBlur={handleBlur("password")}
                      value={values.password}
                      secureTextEntry
                    />
                    {touched.password && errors.password && (
                      <Text className="text-red-500 text-xs mt-1">
                        {errors.password}
                      </Text>
                    )}
                  </View>

                  <TouchableOpacity
                    onPress={handleSubmit}
                    className="bg-[#123458] py-3 rounded-2xl mt-4 shadow-lg shadow-black/50"
                  >
                    <Text className="text-lg text-[#F1EFEC] font-semibold text-center tracking-wide">
                      Sign Up
                    </Text>
                  </TouchableOpacity>
                </View>
              )}
            </Formik>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default Signup;
